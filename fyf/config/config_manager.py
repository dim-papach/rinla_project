"""
Configuration management for FYF

This module provides functionality to generate, validate, and merge configuration files
with command-line arguments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import click
from fyf.config import CosmicConfig, SatelliteConfig, INLAConfig, PlotConfig

class ConfigManager:
    """Manages configuration files and merging with CLI arguments"""
    
    @staticmethod
    def generate_template(output_path: Path) -> None:
        """Generate a template configuration file"""
        template = {
            "simulate": {
                "cosmic_fraction": 0.01,
                "cosmic_value": None,  # Will be converted to NaN
                "cosmic_seed": None,
                "trails": 1,
                "trail_width": 3,
                "min_angle": -45.0,
                "max_angle": 45.0,
                "trail_value": None,  # Will be converted to NaN
                "output_dir": "./output"
            },
            "process": {
                # Basic INLA parameters
                "shape": "none",
                "mesh_cutoff": None,
                "tolerance": 1e-4,
                "restart": 0,
                "scaling": False,
                "nonstationary": False,
                "output_dir": "./processed",
                
                # Mesh parameters
                "mesh_resolution": 30,
                "max_edge_factor": 10.0,
                "outer_edge_factor": 1.5,
                "offset_inner_factor": 0.5,
                "offset_outer_factor": 2.0,
                
                # SPDE parameters
                "alpha": 2,
                "prior_range_prob": 0.2,
                "prior_range_lower": 2.0,
                "prior_sigma_prob": 0.2,
                "prior_sigma_upper": 2.0,
                
                # Computation parameters
                "num_threads": 6,
                "openmp_strategy": "huge",
                
                # Non-stationary parameters
                "nbasis": 2,
                "spline_degree": 10
            },

            "validate": {
                "metrics": ["ssim", "mse", "mae"],
                "generate_plots": True,
                "output_dir": "./validation"
            },
            "plot": {
                "plot_type": "all",
                "dpi": 150,
                "cmap": "viridis",
                "residual_cmap": "viridis",
                "percentile_min": 1,
                "percentile_max": 99,
                "residual_percentile_min": 1,
                "residual_percentile_max": 99,
                "output_dir": "./plots"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        click.echo(f"Configuration template generated: {output_path}")
    
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise click.ClickException(f"Error loading config: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_sections = ["simulate", "process", "validate", "plot"]
        
        for section in required_sections:
            if section not in config:
                click.echo(f"Warning: Missing section '{section}' in config", err=True)
                return False
        
        click.echo("Configuration is valid")
        return True
    
    @staticmethod
    def merge_with_cli_args(config: Dict[str, Any], command: str, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with CLI arguments (CLI args take precedence)"""
        if command not in config:
            config[command] = {}
        
        # CLI arguments override config values
        for key, value in cli_args.items():
            if value is not None:  # Only override if CLI arg was explicitly provided
                config[command][key] = value
        
        return config[command]

    @staticmethod
    def create_configs_from_dict(config_dict: Dict[str, Any]) -> tuple:
        """Create configuration objects from dictionary"""
        simulate_cfg = config_dict.get("simulate", {})
        process_cfg = config_dict.get("process", {})
        plot_cfg = config_dict.get("plot", {})
        
        cosmic_cfg = CosmicConfig(
            fraction=simulate_cfg.get("cosmic_fraction", 0.01),
            value=simulate_cfg.get("cosmic_value", None),
            seed=simulate_cfg.get("cosmic_seed", None)
        )
        
        satellite_cfg = SatelliteConfig(
            num_trails=simulate_cfg.get("trails", 1),
            trail_width=simulate_cfg.get("trail_width", 3),
            min_angle=simulate_cfg.get("min_angle", -45.0),
            max_angle=simulate_cfg.get("max_angle", 45.0),
            value=simulate_cfg.get("trail_value", None)
        )
        
        inla_cfg = INLAConfig(
            # Basic parameters
            shape=process_cfg.get("shape", "none"),
            mesh_cutoff=process_cfg.get("mesh_cutoff", None),
            tolerance=process_cfg.get("tolerance", 1e-4),
            restart=process_cfg.get("restart", 0),
            scaling=bool(process_cfg.get("scaling", False)),
            nonstationary=bool(process_cfg.get("nonstationary", False)),
            
            # Mesh parameters
            mesh_resolution=process_cfg.get("mesh_resolution", 30),
            max_edge_factor=process_cfg.get("max_edge_factor", 10.0),
            outer_edge_factor=process_cfg.get("outer_edge_factor", 1.5),
            offset_inner_factor=process_cfg.get("offset_inner_factor", 0.5),
            offset_outer_factor=process_cfg.get("offset_outer_factor", 2.0),
            
            # SPDE parameters
            alpha=process_cfg.get("alpha", 2),
            prior_range_prob=process_cfg.get("prior_range_prob", 0.2),
            prior_range_lower=process_cfg.get("prior_range_lower", 2.0),
            prior_sigma_prob=process_cfg.get("prior_sigma_prob", 0.2),
            prior_sigma_upper=process_cfg.get("prior_sigma_upper", 2.0),
            
            # Computation parameters
            num_threads=process_cfg.get("num_threads", 6),
            openmp_strategy=process_cfg.get("openmp_strategy", "huge"),
            
            # Non-stationary parameters
            nbasis=process_cfg.get("nbasis", 2),
            spline_degree=process_cfg.get("spline_degree", 10)
        )
        
        plot_config = PlotConfig(
            dpi=plot_cfg.get("dpi", 150),
            cmap=plot_cfg.get("cmap", "viridis"),
            residual_cmap=plot_cfg.get("residual_cmap", "viridis"),
            percentile_range=(plot_cfg.get("percentile_min", 1), plot_cfg.get("percentile_max", 99)),
            residual_percentile=(plot_cfg.get("residual_percentile_min", 1), plot_cfg.get("residual_percentile_max", 99))
        )
        
        return cosmic_cfg, satellite_cfg, inla_cfg, plot_config