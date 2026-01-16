import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from datetime import datetime


def create_optimized_ryg_cmap():
    """Create an optimized Red-Yellow-Green colormap (6-color scale)"""
    colors = [
        (0.00, "#4CAF50"),   # Green
        (0.15, "#81C784"),   # Light Green
        (0.30, "#AED581"),   # Green-Yellow
        (0.45, "#FFF176"),   # Light Yellow
        (0.55, "#FFD54F"),   # Yellow
        (0.70, "#FFB74D"),   # Yellow-Orange
        (0.85, "#FF8A65"),   # Light Red
        (1.00, "#E53935")    # Red
    ]
    return LinearSegmentedColormap.from_list("optimized_ryg", colors)


def parse_matrix_data(log_data):
    """Parse matrix data from log string containing bracketed number sequences"""
    number_sequences = re.findall(r'\[([\d\s]+)\]', log_data)
    return np.array([list(map(int, seq.split())) for seq in number_sequences])


def read_log_file(file_path):
    """Read content from specified log file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise SystemExit(f"Error: File not found - {file_path}")
    except Exception as e:
        raise SystemExit(f"Error reading file: {str(e)}")


def plot_deepxtrace_heatmap(
        matrix,
        title="DeepXTrace Heatmap",
        figsize=(
            15,
            5),
        dpi=100,
        output_format='png',
        cell_ratio=1.5):
    """
    Generate a deepxtrace heatmap

    Args:
        matrix: Input 2D numpy array
        title: Chart title (default: "DeepXTrace Heatmap")
        figsize: Base figure size (will be scaled by cell_ratio)
        dpi: Output resolution in dots per inch (default: 100)
        output_format: Output file format (default: 'png')
        cell_ratio: Cell size scaling factor (default: 1.5)
    """
    # Calculate adjusted figure size based on matrix dimensions
    rows, cols = matrix.shape
    adjusted_figsize = (figsize[0] * cell_ratio * (cols / 10),
                        figsize[1] * cell_ratio * (rows / 10))

    # Configure vector output settings
    plt.figure(figsize=adjusted_figsize)
    plt.rcParams.update({
        'svg.fonttype': 'none',
        'pdf.fonttype': 42
    })

    # Create colormap and normalize data
    cmap = create_optimized_ryg_cmap()
    log_matrix = np.log1p(matrix)
    norm = plt.Normalize(vmin=log_matrix.min(), vmax=log_matrix.max())

    # Dynamic annotation size based on cell size
    annot_size = max(8, min(20, 10 * cell_ratio))

    # Generate heatmap
    ax = sns.heatmap(
        log_matrix,
        cmap=cmap,
        norm=norm,
        annot=matrix,
        fmt='.2e',
        linewidths=0.5,
        linecolor="white",
        annot_kws={
            "size": annot_size,
            "color": "black"
        },
        cbar_kws={
            "label": "Log(Value + 1) Scale",
            "format": ScalarFormatter(),
            "shrink": 0.8
        }
    )

    # Configure labels and title
    ax.set_title(title, fontsize=16 * cell_ratio, pad=20, fontweight='bold')
    plt.xticks(fontsize=10 * cell_ratio, rotation=45)
    plt.yticks(fontsize=10 * cell_ratio)

    # Colorbar customization
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10 * cell_ratio)
    cbar.ax.set_ylabel(
        "Color Scale (Token Wait Time)",
        fontsize=12 * cell_ratio,
        fontweight='bold'
    )

    # Save output
    output_file = f"deepxtrace.{output_format}"
    print(
        f"Saving started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    plt.savefig(
        output_file,
        format=output_format,
        bbox_inches='tight',
        dpi=dpi)
    print(
        f"Saving completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    plt.close()


def main():
    """Command line interface for DeepXTrace heatmap"""
    parser = argparse.ArgumentParser(
        description='Generate DeepXTrace heatmap visualization from log file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'input_file',
        help='Path to log file containing matrix data')
    parser.add_argument(
        '--title',
        default='DeepXTrace Heatmap',
        help='Chart title')
    parser.add_argument('--figsize', nargs=2, type=float, default=[15, 5],
                        help='Base figure dimensions (width height)')
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='Output resolution')
    parser.add_argument(
        '--format',
        default='png',
        choices=[
            'png',
            'svg',
            'pdf'],
        help='Output file format')
    parser.add_argument('--cell_ratio', type=float, default=1.5,
                        help='Cell size scaling factor')

    args = parser.parse_args()

    # Read and process data
    log_content = read_log_file(args.input_file)
    data_matrix = parse_matrix_data(log_content)
    if data_matrix.size == 0:
        raise SystemExit(
            f"Error: No matrix data found in '{args.input_file}'.")

    # Generate visualization
    plot_deepxtrace_heatmap(
        matrix=data_matrix,
        title=args.title,
        figsize=args.figsize,
        dpi=args.dpi,
        output_format=args.format,
        cell_ratio=args.cell_ratio
    )


if __name__ == "__main__":
    main()
