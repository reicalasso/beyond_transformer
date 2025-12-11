#!/usr/bin/env python3
"""
PULSE Command Line Interface - Main Entry Point.

Usage:
    pulse train --config config.yaml
    pulse generate --model path/to/model --prompt "Hello"
    pulse benchmark --task lra --model path/to/model
    pulse convert --input model.pt --output model.onnx
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(
        prog="pulse",
        description="PULSE CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train an PULSE model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    train_parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    train_parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text with PULSE model")
    gen_parser.add_argument("--model", type=str, required=True, help="Path to model")
    gen_parser.add_argument("--prompt", type=str, default="", help="Input prompt")
    gen_parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling")
    gen_parser.add_argument("--num-samples", type=int, default=1, help="Number of samples")
    gen_parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--task", type=str, required=True, 
                              choices=["lra", "babi", "wikitext", "all"],
                              help="Benchmark task")
    bench_parser.add_argument("--model", type=str, required=True, help="Path to model")
    bench_parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    bench_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on dataset")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to model")
    eval_parser.add_argument("--data", type=str, required=True, help="Path to evaluation data")
    eval_parser.add_argument("--task", type=str, default="classification", help="Task type")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    convert_parser.add_argument("--input", type=str, required=True, help="Input model path")
    convert_parser.add_argument("--output", type=str, required=True, help="Output model path")
    convert_parser.add_argument("--format", type=str, default="onnx", 
                                choices=["onnx", "torchscript", "safetensors"],
                                help="Output format")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("--model", type=str, required=True, help="Path to model")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model states")
    viz_parser.add_argument("--model", type=str, required=True, help="Path to model")
    viz_parser.add_argument("--input", type=str, required=True, help="Input text or file")
    viz_parser.add_argument("--output", type=str, default="./visualizations", help="Output directory")
    viz_parser.add_argument("--type", type=str, default="states",
                            choices=["states", "attention", "routing", "all"],
                            help="Visualization type")

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Run training command."""
    from ..models.pulse_lm import PulseConfig, PulseForCausalLM
    from ..training import PulseTrainer, TrainingArguments
    from ..core.config import Config

    logger.info(f"Loading config from {args.config}")
    config = Config(args.config)

    # Create model config
    model_config = PulseConfig(
        vocab_size=config.get("model.vocab_size", 32000),
        hidden_size=config.get("model.hidden_size", 768),
        num_layers=config.get("model.num_layers", 12),
        num_heads=config.get("model.num_heads", 12),
        num_states=config.get("model.num_states", 32),
        state_dim=config.get("model.state_dim", 768),
        intermediate_size=config.get("model.intermediate_size", 3072),
        max_position_embeddings=config.get("model.max_position_embeddings", 2048),
        dropout=config.get("model.dropout", 0.1),
    )

    # Create model
    logger.info("Creating model...")
    model = PulseForCausalLM(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config.get("training.epochs", 3),
        per_device_train_batch_size=config.get("training.batch_size", 8),
        learning_rate=config.get("training.learning_rate", 5e-5),
        warmup_steps=config.get("training.warmup_steps", 0),
        fp16=args.fp16,
        logging_steps=config.get("training.logging_steps", 100),
        save_steps=config.get("training.save_steps", 500),
        seed=args.seed,
        report_to=["wandb"] if args.wandb else ["tensorboard"],
    )

    # Create trainer
    trainer = PulseTrainer(
        model=model,
        args=training_args,
        train_dataset=None,  # Will be loaded from config
        eval_dataset=None,
    )

    # Train
    logger.info("Starting training...")
    result = trainer.train(resume_from_checkpoint=args.resume)

    logger.info(f"Training completed. Best metric: {result.get('best_metric')}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Run generation command."""
    from ..models.pulse_lm import PulseConfig, PulseForCausalLM

    logger.info(f"Loading model from {args.model}")
    
    # Load model
    checkpoint = torch.load(args.model, map_location="cpu")
    
    if "config" in checkpoint:
        config = PulseConfig.from_dict(checkpoint["config"])
    else:
        config = PulseConfig()
    
    model = PulseForCausalLM(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_text(prompt: str) -> str:
        """Generate text from prompt."""
        # Simple tokenization (in practice, use a proper tokenizer)
        input_ids = torch.tensor([[ord(c) % config.vocab_size for c in prompt]], device=device)
        
        output_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )
        
        # Simple detokenization
        output_text = "".join([chr(id % 128) for id in output_ids[0].tolist()])
        return output_text

    if args.interactive:
        logger.info("Interactive mode. Type 'quit' to exit.")
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() == "quit":
                    break
                
                for i in range(args.num_samples):
                    output = generate_text(prompt)
                    print(f"\nGeneration {i+1}:\n{output}")
            except KeyboardInterrupt:
                break
    else:
        for i in range(args.num_samples):
            output = generate_text(args.prompt)
            print(f"\nGeneration {i+1}:\n{output}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark command."""
    logger.info(f"Running {args.task} benchmark...")
    
    os.makedirs(args.output, exist_ok=True)
    
    results = {}
    
    if args.task in ["lra", "all"]:
        logger.info("Running LRA benchmark...")
        # Import and run LRA benchmark
        try:
            from ..benchmarks.lra_benchmark import run_lra_benchmark
            lra_results = run_lra_benchmark(args.model, batch_size=args.batch_size)
            results["lra"] = lra_results
        except ImportError:
            logger.warning("LRA benchmark not available")
    
    if args.task in ["babi", "all"]:
        logger.info("Running bAbI benchmark...")
        try:
            from ..benchmarks.babi_benchmark import run_babi_benchmark
            babi_results = run_babi_benchmark(args.model, batch_size=args.batch_size)
            results["babi"] = babi_results
        except ImportError:
            logger.warning("bAbI benchmark not available")
    
    if args.task in ["wikitext", "all"]:
        logger.info("Running WikiText benchmark...")
        try:
            from ..benchmarks.pg19_benchmark import run_pg19_benchmark
            wikitext_results = run_pg19_benchmark(args.model, batch_size=args.batch_size)
            results["wikitext"] = wikitext_results
        except ImportError:
            logger.warning("WikiText benchmark not available")
    
    # Save results
    output_file = os.path.join(args.output, "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    for task, task_results in results.items():
        print(f"\n{task.upper()}:")
        if isinstance(task_results, dict):
            for metric, value in task_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show model information."""
    from ..models.pulse_lm import PulseConfig, PulseForCausalLM

    logger.info(f"Loading model from {args.model}")
    
    checkpoint = torch.load(args.model, map_location="cpu")
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    # Config info
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Parameter info
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Layer breakdown
    print("\nLayer breakdown:")
    layer_params = {}
    for name, param in state_dict.items():
        layer_type = name.split(".")[0]
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()
    
    for layer, count in sorted(layer_params.items(), key=lambda x: -x[1]):
        percentage = 100 * count / total_params
        print(f"  {layer}: {count:,} ({percentage:.1f}%)")
    
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert model format."""
    from ..models.pulse_lm import PulseConfig, PulseForCausalLM

    logger.info(f"Converting {args.input} to {args.format}")
    
    # Load model
    checkpoint = torch.load(args.input, map_location="cpu")
    
    if "config" in checkpoint:
        config = PulseConfig.from_dict(checkpoint["config"])
    else:
        config = PulseConfig()
    
    model = PulseForCausalLM(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    if args.format == "onnx":
        # Export to ONNX
        dummy_input = torch.randint(0, config.vocab_size, (1, 128))
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
        )
        logger.info(f"Exported to ONNX: {args.output}")
    
    elif args.format == "torchscript":
        # Export to TorchScript
        dummy_input = torch.randint(0, config.vocab_size, (1, 128))
        traced = torch.jit.trace(model, dummy_input)
        traced.save(args.output)
        logger.info(f"Exported to TorchScript: {args.output}")
    
    elif args.format == "safetensors":
        # Export to safetensors
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), args.output)
            logger.info(f"Exported to safetensors: {args.output}")
        except ImportError:
            logger.error("safetensors not installed. Run: pip install safetensors")
            return 1
    
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize model states."""
    logger.info("Visualization command - see pulse.visualization module for details")
    
    os.makedirs(args.output, exist_ok=True)
    
    # This would call the visualization module
    logger.info(f"Visualizations will be saved to {args.output}")
    logger.info("Use the Python API for detailed visualization:")
    logger.info("  from pulse.visualization import StateVisualizer")
    logger.info("  viz = StateVisualizer(model)")
    logger.info("  viz.plot_states(input_text)")
    
    return 0


def app() -> int:
    """Main application entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "train": cmd_train,
        "generate": cmd_generate,
        "benchmark": cmd_benchmark,
        "info": cmd_info,
        "convert": cmd_convert,
        "visualize": cmd_visualize,
    }

    handler = commands.get(args.command)
    if handler is None:
        logger.error(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except Exception as e:
        logger.exception(f"Error running command: {e}")
        return 1


def main() -> None:
    """Main entry point."""
    sys.exit(app())


if __name__ == "__main__":
    main()
