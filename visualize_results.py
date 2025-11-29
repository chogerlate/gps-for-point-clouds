import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct


def read_ply(ply_path):
    """
    Read a PLY file and extract vertex coordinates.
    Supports ASCII and binary PLY formats.
    """
    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        line = f.readline().decode('ascii').strip()
        
        if line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        header_lines.append(line)
        
        vertex_count = 0
        format_type = None
        properties = []
        in_vertex_element = False
        
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith('element'):
                in_vertex_element = False
            elif line.startswith('property') and in_vertex_element:
                properties.append(line.split()[-1])
            elif line == 'end_header':
                break
        
        # Find x, y, z indices
        try:
            x_idx = properties.index('x')
            y_idx = properties.index('y')
            z_idx = properties.index('z')
        except ValueError:
            raise ValueError("PLY file doesn't contain x, y, z coordinates")
        
        # Read vertex data
        if format_type == 'ascii':
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                values = line.split()
                points.append([float(values[x_idx]), float(values[y_idx]), float(values[z_idx])])
            points = np.array(points)
        else:
            # Binary format - read all properties but extract only x,y,z
            dtype_map = {'float': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint': 'u4', 'int': 'i4'}
            dtype_list = []
            for prop_line in [h for h in header_lines if h.startswith('property') and any(h.endswith(p) for p in properties)]:
                parts = prop_line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                dtype_list.append((prop_name, dtype_map.get(prop_type, 'f4')))
            
            dt = np.dtype(dtype_list)
            data = np.frombuffer(f.read(dt.itemsize * vertex_count), dtype=dt, count=vertex_count)
            points = np.column_stack([data['x'], data['y'], data['z']])
    
    return points


def read_xyz(xyz_path):
    """
    Read an XYZ file (simple text format with x y z per line).
    """
    return np.loadtxt(xyz_path, delimiter=None)[:, :3]  # Take only first 3 columns


def read_point_cloud(file_path):
    """
    Auto-detect and read point cloud file (.ply or .xyz).
    """
    if file_path.endswith('.ply'):
        return read_ply(file_path)
    elif file_path.endswith('.xyz'):
        return read_xyz(file_path)
    else:
        raise ValueError("Unsupported file format. Use .ply or .xyz")


def visualize_point_cloud(file_path, title="Point Cloud", point_size=1, 
                          color_by='z', colormap='viridis', alpha=0.6,
                          elev=30, azim=45, figsize=(10, 8), 
                          subsample=None, hide_axes=False):
    """
    Visualize a 3D point cloud from a PLY or XYZ file.
    
    Parameters:
    -----------
    file_path : str
        Path to the .ply or .xyz file
    title : str
        Title for the plot
    point_size : float
        Size of the points in the scatter plot
    color_by : str
        Color points by coordinate ('x', 'y', 'z') or 'uniform'
    colormap : str
        Matplotlib colormap name
    alpha : float
        Transparency of points (0-1)
    elev : float
        Elevation angle for viewing
    azim : float
        Azimuth angle for viewing
    figsize : tuple
        Figure size (width, height)
    subsample : int or None
        If set, randomly subsample to this many points for faster rendering
    hide_axes : bool
        If True, hide the axes
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    point_count : int
        Number of points in the cloud (before subsampling)
    """
    # Read point cloud file
    points = read_point_cloud(file_path)
    total_count = len(points)
    
    # Subsample if requested
    if subsample and total_count > subsample:
        idx = np.random.choice(total_count, subsample, replace=False)
        points = points[idx]
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    point_count = len(points)
    
    print(f"Loaded {total_count:,} points, displaying {point_count:,} points")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine coloring
    if color_by == 'x':
        colors = x
    elif color_by == 'y':
        colors = y
    elif color_by == 'z':
        colors = z
    elif color_by == 'uniform':
        colors = 'blue'
    else:
        colors = z  # default
    
    # Plot points
    scatter = ax.scatter(x, y, z, c=colors, cmap=colormap, 
                        s=point_size, alpha=alpha, marker='.')
    
    # Add colorbar if not uniform color
    if color_by != 'uniform':
        plt.colorbar(scatter, ax=ax, label=f'{color_by.upper()} coordinate', 
                     shrink=0.5, aspect=5)
    
    # Set labels and title
    if not hide_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_axis_off()
    
    display_text = f'{title}\n({total_count:,} points'
    if subsample and total_count > subsample:
        display_text += f', showing {point_count:,}'
    display_text += ')'
    ax.set_title(display_text, fontsize=12, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Make axes equal for proper aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    return fig, ax, total_count


def compare_point_clouds(original_path, simplified_path, 
                         point_size=1, colormap='viridis', 
                         figsize=(16, 7), subsample=10000,
                         hide_axes=False):
    """
    Compare original and simplified point clouds side by side.
    
    Parameters:
    -----------
    original_path : str
        Path to the original .ply or .xyz file
    simplified_path : str
        Path to the simplified .ply or .xyz file
    point_size : float
        Size of the points
    colormap : str
        Matplotlib colormap name
    figsize : tuple
        Figure size (width, height)
    subsample : int or None
        Subsample both clouds to this many points for faster rendering
    hide_axes : bool
        If True, hide the axes
    
    Returns:
    --------
    fig : matplotlib figure object
    reduction_ratio : float
        Percentage of points removed
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Load both point clouds
    points_orig = read_point_cloud(original_path)
    points_simp = read_point_cloud(simplified_path)
    
    count_orig = len(points_orig)
    count_simp = len(points_simp)
    reduction = (1 - count_simp / count_orig) * 100
    
    print(f"Original: {count_orig:,} points")
    print(f"Simplified: {count_simp:,} points")
    print(f"Reduction: {reduction:.2f}%")
    
    # Subsample if requested
    if subsample:
        if count_orig > subsample:
            idx = np.random.choice(count_orig, subsample, replace=False)
            points_orig = points_orig[idx]
        if count_simp > subsample:
            idx = np.random.choice(count_simp, subsample, replace=False)
            points_simp = points_simp[idx]
    
    # Extract coordinates
    x1, y1, z1 = points_orig[:, 0], points_orig[:, 1], points_orig[:, 2]
    x2, y2, z2 = points_simp[:, 0], points_simp[:, 1], points_simp[:, 2]
    
    # Plot original
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(x1, y1, z1, c=z1, cmap=colormap, 
                          s=point_size, alpha=0.6, marker='.')
    if hide_axes:
        ax1.set_axis_off()
    else:
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
    ax1.set_title(f'Original\n({count_orig:,} points)', 
                  fontsize=12, fontweight='bold')
    
    # Plot simplified
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(x2, y2, z2, c=z2, cmap=colormap, 
                          s=point_size, alpha=0.6, marker='.')
    if hide_axes:
        ax2.set_axis_off()
    else:
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    ax2.set_title(f'Simplified\n({count_simp:,} points, {reduction:.1f}% reduction)', 
                  fontsize=12, fontweight='bold')
    
    # Match viewing angles and axes limits
    for ax in [ax1, ax2]:
        ax.view_init(elev=30, azim=45)
        all_x = np.concatenate([x1, x2])
        all_y = np.concatenate([y1, y2])
        all_z = np.concatenate([z1, z2])
        max_range = np.array([all_x.max()-all_x.min(), 
                             all_y.max()-all_y.min(), 
                             all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    return fig, reduction


def visualize_point_cloud_interactive(file_path, title="Point Cloud", 
                                      point_size=2, subsample=None):
    """
    Create an interactive Plotly visualization of a point cloud.
    
    Parameters:
    -----------
    file_path : str
        Path to the .ply or .xyz file
    title : str
        Title for the plot
    point_size : float
        Size of the points
    subsample : int or None
        If set, randomly subsample to this many points
    
    Returns:
    --------
    fig : plotly figure object
    point_count : int
        Total number of points
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return None, 0
    
    # Read point cloud
    points = read_point_cloud(file_path)
    total_count = len(points)
    
    # Subsample if requested
    if subsample and total_count > subsample:
        idx = np.random.choice(total_count, subsample, replace=False)
        points = points[idx]
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    point_count = len(points)
    
    print(f"Loaded {total_count:,} points, displaying {point_count:,} points")
    
    # Create interactive plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=point_size,
            color=z,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Z")
        )
    )])
    
    display_text = f'{title} ({total_count:,} points'
    if subsample and total_count > subsample:
        display_text += f', showing {point_count:,}'
    display_text += ')'
    
    fig.update_layout(
        title=display_text,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig, total_count


def compare_point_clouds_interactive(original_path, simplified_path,
                                     point_size=2, subsample=10000):
    """
    Create an interactive side-by-side comparison using Plotly.
    
    Parameters:
    -----------
    original_path : str
        Path to the original file
    simplified_path : str
        Path to the simplified file
    point_size : float
        Size of points
    subsample : int or None
        Subsample both clouds to this many points
    
    Returns:
    --------
    fig : plotly figure object
    reduction_ratio : float
        Percentage of points removed
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return None, 0
    
    # Load both point clouds
    points_orig = read_point_cloud(original_path)
    points_simp = read_point_cloud(simplified_path)
    
    count_orig = len(points_orig)
    count_simp = len(points_simp)
    reduction = (1 - count_simp / count_orig) * 100
    
    print(f"Original: {count_orig:,} points")
    print(f"Simplified: {count_simp:,} points")
    print(f"Reduction: {reduction:.2f}%")
    
    # Subsample if requested
    if subsample:
        if count_orig > subsample:
            idx = np.random.choice(count_orig, subsample, replace=False)
            points_orig = points_orig[idx]
        if count_simp > subsample:
            idx = np.random.choice(count_simp, subsample, replace=False)
            points_simp = points_simp[idx]
    
    x1, y1, z1 = points_orig[:, 0], points_orig[:, 1], points_orig[:, 2]
    x2, y2, z2 = points_simp[:, 0], points_simp[:, 1], points_simp[:, 2]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(f'Original<br>({count_orig:,} points)', 
                       f'Simplified<br>({count_simp:,} points, {reduction:.1f}% reduction)')
    )
    
    # Add original
    fig.add_trace(
        go.Scatter3d(
            x=x1, y=y1, z=z1,
            mode='markers',
            marker=dict(size=point_size, color=z1, colorscale='Viridis', opacity=0.8),
            name='Original'
        ),
        row=1, col=1
    )
    
    # Add simplified
    fig.add_trace(
        go.Scatter3d(
            x=x2, y=y2, z=z2,
            mode='markers',
            marker=dict(size=point_size, color=z2, colorscale='Viridis', opacity=0.8),
            name='Simplified'
        ),
        row=1, col=2
    )
    
    # Update layout for both subplots
    fig.update_layout(
        height=600,
        showlegend=False,
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data')
    )
    
    return fig, reduction


# Example usage:
if __name__ == "__main__":
    # Interactive single visualization
    # fig, count = visualize_point_cloud_interactive(
    #     'point_cloud.xyz',
    #     title="My Point Cloud",
    #     point_size=2,
    #     subsample=10000
    # )
    # fig.show()
    
    # Interactive comparison
    # fig, reduction = compare_point_clouds_interactive(
    #     'original.xyz',
    #     'simplified.xyz',
    #     point_size=2,
    #     subsample=10000
    # )
    # fig.show()
    
    pass