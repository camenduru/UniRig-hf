import shutil
import subprocess
import time
from pathlib import Path
from typing import Tuple

import gradio as gr
import lightning as L
import torch
import yaml
from box import Box

# Get the PyTorch and CUDA versions
torch_version = torch.__version__.split("+")[0]  # Strips any "+cuXXX" suffix
cuda_version = torch.version.cuda
spconv_version = "-cu121" if cuda_version else "" 

# Format CUDA version to match the URL convention (e.g., "cu118" for CUDA 11.8)
if cuda_version:
    cuda_version = f"cu{cuda_version.replace('.', '')}"
else:
    cuda_version = "cpu"  # Fallback in case CUDA is not available

subprocess.run(f'pip install spconv{spconv_version}', shell=True)
subprocess.run(f'pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html --no-cache-dir', shell=True)

# Helper functions
def validate_input_file(file_path: str) -> bool:
    """Validate if the input file format is supported."""
    supported_formats = ['.obj', '.fbx', '.glb']
    if not file_path or not Path(file_path).exists():
        return False
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats

def extract_mesh_python(input_file: str, output_dir: str) -> str:
    """
    Extract mesh data from 3D model using Python (replaces extract.sh)
    Returns path to generated .npz file
    """
    # Import required modules
    from src.data.extract import extract_builtin, get_files
    
    # Create extraction parameters
    files = get_files(
        data_name="raw_data.npz",
        inputs=str(input_file),
        input_dataset_dir=None,
        output_dataset_dir=output_dir,
        force_override=True,
        warning=False,
    )
    
    if not files:
        raise RuntimeError("No files to extract")
    
    # Run the actual extraction
    timestamp = str(int(time.time()))
    extract_builtin(
        output_folder=output_dir,
        target_count=50000,
        num_runs=1,
        id=0,
        time=timestamp,
        files=files,
    )
    
    # Return the directory path where raw_data.npz was created
    # The dataset expects to find raw_data.npz in this directory
    expected_npz_dir = files[0][1]  # This is the output directory
    expected_npz_file = Path(expected_npz_dir) / "raw_data.npz"
    
    if not expected_npz_file.exists():
        raise RuntimeError(f"Extraction failed: {expected_npz_file} not found")
    
    return expected_npz_dir  # Return the directory containing raw_data.npz

def run_inference_python(
    input_file: str, 
    output_file: str, 
    inference_type: str, 
    seed: int = 12345, 
    npz_dir: str = None
) -> str:
    """
    Unified inference function for both skeleton and skin inference.
    
    Args:
        input_file: Path to input file (3D model for skeleton, skeleton FBX for skin)
        output_file: Path to output file
        inference_type: Either "skeleton" or "skin"
        seed: Random seed for reproducible results
        npz_dir: Directory for NPZ files (used for skeleton inference)
    
    Returns:
        Path to generated file
    """
    from src.data.datapath import Datapath
    from src.data.dataset import DatasetConfig, UniRigDatasetModule
    from src.data.transform import TransformConfig
    from src.inference.download import download
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    # Set random seed for skeleton inference
    if inference_type == "skeleton":
        L.seed_everything(seed, workers=True)
    
    # Load task and model configurations based on inference type
    if inference_type == "skeleton":
        task_config_path = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
        transform_config_path = "configs/transform/inference_ar_transform.yaml"
        model_config_path = "configs/model/unirig_ar_350m_1024_81920_float32.yaml"
        system_config_path = "configs/system/ar_inference_articulationxl.yaml"
        tokenizer_config_path = "configs/tokenizer/tokenizer_parts_articulationxl_256.yaml"
        data_name = "raw_data.npz"
    else:  # skin
        task_config_path = "configs/task/quick_inference_unirig_skin.yaml"
        transform_config_path = "configs/transform/inference_skin_transform.yaml"
        model_config_path = "configs/model/unirig_skin.yaml"
        system_config_path = "configs/system/skin.yaml"
        tokenizer_config_path = None
        data_name = "predict_skeleton.npz"
    
    # Load task configuration
    if not Path(task_config_path).exists():
        raise FileNotFoundError(f"Task configuration file not found: {task_config_path}")
    
    with open(task_config_path, 'r') as f:
        task = Box(yaml.safe_load(f))
    
    # Setup data directory and datapath
    if inference_type == "skeleton":
        # Create temporary npz directory and extract mesh data
        if npz_dir is None:
            npz_dir = Path(output_file).parent / "npz"
        npz_dir = Path(npz_dir)
        npz_dir.mkdir(exist_ok=True)
        npz_data_dir = extract_mesh_python(input_file, npz_dir)
        datapath = Datapath(files=[npz_data_dir], cls=None)
    else:  # skin
        # Look for NPZ files from previous skeleton inference
        skeleton_work_dir = Path(input_file).parent
        all_npz_files = list(skeleton_work_dir.rglob("**/*.npz"))
        if not all_npz_files:
            raise RuntimeError(f"No NPZ files found for skin inference in {skeleton_work_dir}")
        skeleton_npz_dir = all_npz_files[0].parent
        datapath = Datapath(files=[str(skeleton_npz_dir)], cls=None)
    
    # Load common configurations
    data_config = Box(yaml.safe_load(open("configs/data/quick_inference.yaml", 'r')))
    transform_config = Box(yaml.safe_load(open(transform_config_path, 'r')))
    
    # Setup tokenizer and model
    if inference_type == "skeleton":
        tokenizer_config = TokenizerConfig.parse(config=Box(yaml.safe_load(open(tokenizer_config_path, 'r'))))
        tokenizer = get_tokenizer(config=tokenizer_config)
        model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
        model = get_model(tokenizer=tokenizer, **model_config)
    else:  # skin
        tokenizer_config = None
        tokenizer = None
        model_config = Box(yaml.safe_load(open(model_config_path, 'r')))
        model = get_model(tokenizer=None, **model_config)
    
    # Setup datasets and transforms
    predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
    predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)
    
    # Create data module
    data = UniRigDatasetModule(
        process_fn=model._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name,
        datapath=datapath,
        cls=None,
    )
    
    # Setup callbacks and writer
    callbacks = []
    writer_config = task.writer.copy()
    
    if inference_type == "skeleton":
        writer_config['npz_dir'] = str(npz_dir)
        writer_config['output_dir'] = str(Path(output_file).parent)
        writer_config['output_name'] = Path(output_file).name
        writer_config['user_mode'] = False  # Enable NPZ export for skeleton
    else:  # skin
        writer_config['npz_dir'] = str(skeleton_npz_dir)
        writer_config['output_name'] = str(output_file)
        writer_config['user_mode'] = True
        writer_config['export_fbx'] = True
    
    print(f"Writer config for {inference_type}: {writer_config}")
    callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
    
    # Get system
    system_config = Box(yaml.safe_load(open(system_config_path, 'r')))
    system = get_system(**system_config, model=model, steps_per_epoch=1)
    
    # Setup trainer
    trainer_config = task.trainer
    resume_from_checkpoint = download(task.resume_from_checkpoint)
    
    trainer = L.Trainer(callbacks=callbacks, logger=None, **trainer_config)
    
    # Run prediction
    trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
    
    # Handle output file location and validation
    if inference_type == "skeleton":
        # Look for the generated skeleton.fbx file
        input_name_stem = Path(input_file).stem
        actual_output_dir = Path(output_file).parent / input_name_stem
        actual_output_file = actual_output_dir / "skeleton.fbx"
        
        if not actual_output_file.exists():
            # Try alternative locations
            alt_files = list(Path(output_file).parent.rglob("skeleton.fbx"))
            if alt_files:
                actual_output_file = alt_files[0]
                print(f"Found skeleton at alternative location: {actual_output_file}")
            else:
                all_files = list(Path(output_file).parent.rglob("*"))
                print(f"Available files: {[str(f) for f in all_files]}")
                raise RuntimeError(f"Skeleton FBX file not found. Expected at: {actual_output_file}")
        
        # Copy to the expected output location
        if actual_output_file != Path(output_file):
            shutil.copy2(actual_output_file, output_file)
            print(f"Copied skeleton from {actual_output_file} to {output_file}")
    
    else:  # skin
        # Check if skin FBX file was generated
        if not Path(output_file).exists():
            # Look for generated skin FBX files
            skin_files = list(Path(output_file).parent.rglob("*skin*.fbx"))
            if skin_files:
                actual_output_file = skin_files[0]
                shutil.copy2(actual_output_file, output_file)
            else:
                raise RuntimeError(f"Skin FBX file not found. Expected at: {output_file}")
    
    print(f"Generated {inference_type} at: {output_file}")
    return str(output_file)

def merge_results_python(source_file: str, target_file: str, output_file: str) -> str:
    """
    Merge results using Python (replaces merge.sh)
    Returns path to merged file
    """
    from src.inference.merge import transfer
    
    # Validate input paths
    if not Path(source_file).exists():
        raise ValueError(f"Source file does not exist: {source_file}")
    if not Path(target_file).exists():
        raise ValueError(f"Target file does not exist: {target_file}")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the transfer function directly
    transfer(source=str(source_file), target=str(target_file), output=str(output_path), add_root=False)
    
    # Validate that the output file was created and is a valid file
    if not output_path.exists():
        raise RuntimeError(f"Merge failed: Output file not created at {output_path}")
    
    if not output_path.is_file():
        raise RuntimeError(f"Merge failed: Output path is not a valid file: {output_path}")
    
    return str(output_path.resolve())

def main(input_file: str, seed: int = 12345) -> Tuple[str, list]:
    """
    Run the rigging pipeline based on selected mode.
    
    Args:
        input_file: Path to the input 3D model file
        seed: Random seed for reproducible results
        
    Returns:
        Tuple of (final_file_path, list_of_intermediate_files)
    """
    # Create temp directory
    base_dir = Path(__file__).parent
    temp_dir = base_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    
    # Supported file formats
    supported_formats = ['.obj', '.fbx', '.glb']
    
    # Validate input file
    if not validate_input_file(input_file):
        raise gr.Error(f"Error: Invalid or unsupported file format. Supported formats: {', '.join(supported_formats)}")
    
    # Create working directory
    file_stem = Path(input_file).stem
    input_model_dir = temp_dir / f"{file_stem}_{seed}"
    input_model_dir.mkdir(exist_ok=True)

    # Copy input file to working directory
    input_file = Path(input_file)
    shutil.copy2(input_file, input_model_dir / input_file.name)
    input_file = input_model_dir / input_file.name
    print(f"New input file path: {input_file}")
    
    # Initialize file paths and output list
    output_files = []
    final_file = None
    
    # Step 1: Generate skeleton
    intermediate_skeleton_file = input_model_dir / f"{file_stem}_skeleton.fbx"
    final_skeleton_file = input_model_dir / f"{file_stem}_skeleton_only{input_file.suffix}"
    run_inference_python(input_file, intermediate_skeleton_file, "skeleton", seed)
    merge_results_python(intermediate_skeleton_file, input_file, final_skeleton_file)
    
    # Step 2: Generate skinning and Merge everything together
    intermediate_skin_file = input_model_dir / f"{file_stem}_skin.fbx"
    final_skin_file = input_model_dir / f"{file_stem}_skeleton_and_skinning{input_file.suffix}"
    run_inference_python(intermediate_skeleton_file, intermediate_skin_file, "skin")
    merge_results_python(intermediate_skin_file, input_file, final_skin_file)
    
    final_file = str(final_skin_file)
    output_files = [str(final_skeleton_file), str(final_skin_file)]

    return final_file, output_files

def create_app():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="UniRig - 3D Model Rigging Demo") as interface:
        
        # Header
        gr.HTML("""
        <div class="title" style="text-align: center">
            <h1>🎯 UniRig: Automated 3D Model Rigging</h1>
            <p style="font-size: 1.1em; color: #6b7280;">
                Leverage deep learning to automatically generate skeletons and skinning weights for your 3D models
            </p>
        </div>
        """)
        
        # Usage Instructions Section
        gr.Markdown("""## Notes:
- If you are not seeing the 3D model preview and you are using chrome, go to `chrome://flags/#enable-unsafe-webgpu` and enable the flag.
- Supported File Formats are `.obj`, `.fbx`, `.glb`
- The process may take a few minutes depending on the model complexity and server load.
        """)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_3d_model = gr.Model3D(label="Upload 3D Model")
                
                with gr.Group():
                    with gr.Row(equal_height=True):
                        seed = gr.Number(
                            value=int(torch.randint(0, 100000, (1,)).item()),
                            label="Random Seed (for reproducible results)",
                            scale=4,
                        )
                        random_btn = gr.Button("🔄 Random Seed", variant="secondary", scale=1)
                
                pipeline_btn = gr.Button("🎯 Start Processing", variant="primary", size="lg")
            
            with gr.Column():
                pipeline_skeleton_out = gr.Model3D(label="Final Result", scale=4)
                files_to_download = gr.Files(label="Download Files", scale=1)
                   
        random_btn.click(
            fn=lambda: int(torch.randint(0, 100000, (1,)).item()),
            outputs=seed
        )
        
        pipeline_btn.click(
            fn=main,
            inputs=[input_3d_model, seed],
            outputs=[pipeline_skeleton_out, files_to_download]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2em; padding: 1em; border-radius: 8px;">
            <p style="color: #6b7280;">
                🔬 <strong>UniRig</strong> - Research by Tsinghua University & Tripo<br>
                📄 <a href="https://arxiv.org/abs/2504.12451" target="_blank">Paper</a> | 
                🏠 <a href="https://zjp-shadow.github.io/works/UniRig/" target="_blank">Project Page</a> | 
                🤗 <a href="https://huggingface.co/VAST-AI/UniRig" target="_blank">Models</a>
            </p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    app = create_app()
    
    # Launch configuration
    app.queue().launch(share=True)
