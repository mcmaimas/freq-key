"""
Frequency-Based Keyboard Layout Module

Contains data structures, generation algorithms, and visualization functions
for creating frequency-based keyboard layouts.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================================================
# Configuration Constants
# =============================================================================

BASE_WIDTH = 7.0      # mm
BASE_HEIGHT = 10.0    # mm
PADDING = 1.0         # mm between keys
ROW_OFFSET = 4.0      # mm horizontal offset per row

MIN_WIDTH = 3.0       # mm minimum key width
MIN_HEIGHT = 5.0      # mm minimum key height

MIN_SCALE = max(MIN_WIDTH / BASE_WIDTH, MIN_HEIGHT / BASE_HEIGHT)


# =============================================================================
# Character Frequency Data (English)
# =============================================================================

FREQUENCIES = {
    # Row 1
    'Q': 0.10, 'W': 2.4, 'E': 12.7, 'R': 6.0, 'T': 9.1,
    'Y': 2.0, 'U': 2.8, 'I': 7.0, 'O': 7.5, 'P': 1.9,
    # Row 2
    'A': 8.2, 'S': 6.3, 'D': 4.3, 'F': 2.2, 'G': 2.0,
    'H': 6.1, 'J': 0.15, 'K': 0.8, 'L': 4.0,
    # Row 3
    'Z': 0.07, 'X': 0.15, 'C': 2.8, 'V': 1.0, 'B': 1.5,
    'N': 6.7, 'M': 2.4
}

ROWS = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],  # Row 0
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],        # Row 1
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']                   # Row 2
]


# =============================================================================
# Key Data Structure
# =============================================================================

@dataclass
class Key:
    """Represents a keyboard key with position and dimensions."""
    letter: str
    x: float          # Left edge position
    y: float          # Bottom edge position
    width: float
    height: float
    frequency: float
    scale: float      # Scale relative to uniform (>1 means larger than baseline)
    
    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y + self.height
    
    @property
    def bottom(self) -> float:
        return self.y
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class GridKey:
    """
    Represents a keyboard key as a collection of grid columns.
    Each column can have a different height, allowing for non-rectangular shapes.
    """
    letter: str
    frequency: float
    row: int                                    # Which keyboard row (0, 1, 2)
    column_bounds: Dict[int, Tuple[float, float]]  # col -> (top_y, bottom_y)
    
    @property
    def area(self) -> float:
        """Calculate total area of the key."""
        total = 0
        for col, (top, bottom) in self.column_bounds.items():
            total += abs(bottom - top)  # 1mm column width × height
        return total
    
    @property
    def uniform_area(self) -> float:
        """Area of a uniform key (7mm × 10mm)."""
        return BASE_WIDTH * BASE_HEIGHT
    
    @property
    def scale(self) -> float:
        """Scale relative to uniform key."""
        return self.area / self.uniform_area if self.uniform_area > 0 else 1.0
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculate centroid of the key shape."""
        if not self.column_bounds:
            return (0, 0)
        total_area = 0
        cx_sum = 0
        cy_sum = 0
        for col, (top, bottom) in self.column_bounds.items():
            h = abs(bottom - top)
            cx_sum += (col + 0.5) * h
            cy_sum += ((top + bottom) / 2) * h
            total_area += h
        if total_area == 0:
            return (0, 0)
        return (cx_sum / total_area, cy_sum / total_area)
    
    def get_polygon_points(self, h_padding: float = 0.5) -> List[Tuple[float, float]]:
        """
        Get polygon vertices for rendering (clockwise from top-left).
        
        Args:
            h_padding: Horizontal padding to apply on left/right edges (default 0.5mm)
        """
        if not self.column_bounds:
            return []
        
        cols = sorted(self.column_bounds.keys())
        first_col = cols[0]
        last_col = cols[-1]
        
        # Top edge: left to right (with horizontal padding)
        top_points = []
        for col in cols:
            x = col
            if col == first_col:
                x += h_padding  # Left edge padding
            top_points.append((x, self.column_bounds[col][0]))
        
        # Right edge (with padding)
        top_points.append((last_col + 1 - h_padding, self.column_bounds[last_col][0]))
        
        # Bottom edge: right to left (with horizontal padding)
        bottom_points = [(last_col + 1 - h_padding, self.column_bounds[last_col][1])]
        for col in reversed(cols):
            x = col
            if col == first_col:
                x += h_padding  # Left edge padding
            bottom_points.append((x, self.column_bounds[col][1]))
        
        return top_points + bottom_points


# =============================================================================
# Keyboard Generation
# =============================================================================

def generate_uniform_keyboard() -> List[Key]:
    """Generate a traditional keyboard with uniform key sizes."""
    keys = []
    row_pitch = BASE_HEIGHT + PADDING
    
    for row_idx, row_keys in enumerate(ROWS):
        x = row_idx * ROW_OFFSET
        y = -row_idx * row_pitch
        
        for letter in row_keys:
            keys.append(Key(
                letter=letter,
                x=x,
                y=y,
                width=BASE_WIDTH,
                height=BASE_HEIGHT,
                frequency=FREQUENCIES[letter],
                scale=1.0
            ))
            x += BASE_WIDTH + PADDING
    
    return keys


def generate_keyboard(scale_method: str = 'linear', enforce_minimum: bool = True) -> List[Key]:
    """
    Generate keyboard with frequency-based width redistribution.
    
    Total width per row stays constant (same as uniform layout).
    Width is redistributed based on frequency — frequent keys grow, rare keys shrink.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log' - how to map frequency to weight
        enforce_minimum: Whether to enforce minimum key size
    
    Returns:
        List of Key objects
    """
    keys = []
    row_pitch = BASE_HEIGHT + PADDING
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        total_key_width = n * BASE_WIDTH
        
        # Get frequencies for this row
        row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        # Apply scaling method
        if scale_method == 'linear':
            weights = {k: v for k, v in row_freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in row_freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
        else:
            weights = {k: v for k, v in row_freqs.items()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: w / total_weight for k, w in weights.items()}
        
        # Calculate widths
        widths = {k: normalized_weights[k] * total_key_width for k in row_keys}
        
        # Enforce minimum width
        if enforce_minimum:
            deficit = 0
            for letter in row_keys:
                if widths[letter] < MIN_WIDTH:
                    deficit += MIN_WIDTH - widths[letter]
                    widths[letter] = MIN_WIDTH
            
            if deficit > 0:
                large_keys = [k for k in row_keys if widths[k] > MIN_WIDTH]
                large_total = sum(widths[k] for k in large_keys)
                for letter in large_keys:
                    reduction = deficit * (widths[letter] / large_total)
                    widths[letter] -= reduction
        
        # Calculate scale relative to uniform
        scales = {k: widths[k] / BASE_WIDTH for k in row_keys}
        
        # Position keys left-to-right
        x = row_idx * ROW_OFFSET
        y = -row_idx * row_pitch
        
        for letter in row_keys:
            keys.append(Key(
                letter=letter,
                x=x,
                y=y,
                width=widths[letter],
                height=BASE_HEIGHT,
                frequency=FREQUENCIES[letter],
                scale=scales[letter]
            ))
            x += widths[letter] + PADDING
    
    return keys


def generate_keyboard_height(scale_method: str = 'linear', enforce_minimum: bool = True) -> List[Key]:
    """
    Generate keyboard with frequency-based height-only scaling.
    
    Width stays uniform. Each key has individual height based on frequency.
    Heights redistributed within each row. Row boundaries are fixed.
    Keys align by TOP edge. Heights capped to prevent overflow.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
        enforce_minimum: Whether to enforce minimum key size
    """
    # Fixed row positions (same as uniform keyboard)
    row_pitch = BASE_HEIGHT + PADDING
    row_tops = [0, -row_pitch, -2 * row_pitch]
    max_height_per_row = BASE_HEIGHT  # Keys can't extend beyond their row slot
    
    keys = []
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        total_height_budget = n * BASE_HEIGHT  # Redistribute this among keys
        
        row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        # Apply scaling method
        if scale_method == 'linear':
            weights = {k: v for k, v in row_freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in row_freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
        else:
            weights = {k: v for k, v in row_freqs.items()}
        
        # Normalize and calculate heights
        total_weight = sum(weights.values())
        heights = {k: (weights[k] / total_weight) * total_height_budget for k in row_keys}
        
        # Cap heights at max (to prevent overflow into next row)
        for letter in row_keys:
            heights[letter] = min(heights[letter], max_height_per_row)
        
        # Enforce minimum height
        if enforce_minimum:
            for letter in row_keys:
                heights[letter] = max(heights[letter], MIN_HEIGHT)
        
        # Position keys (aligned by TOP edge)
        x = row_idx * ROW_OFFSET
        row_top = row_tops[row_idx]
        
        for letter in row_keys:
            h = heights[letter]
            keys.append(Key(
                letter=letter,
                x=x,
                y=row_top - h,  # TOP aligned
                width=BASE_WIDTH,
                height=h,
                frequency=FREQUENCIES[letter],
                scale=h / BASE_HEIGHT
            ))
            x += BASE_WIDTH + PADDING
    
    return keys


def generate_keyboard_2d(scale_method: str = 'linear', enforce_minimum: bool = True) -> List[Key]:
    """
    Generate keyboard with frequency-based width AND height scaling.
    
    Width: redistributed within each row (row width constant).
    Height: individual per key, redistributed within row, capped to prevent overflow.
    Row boundaries are fixed. Keys align by TOP edge.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
        enforce_minimum: Whether to enforce minimum key size
    """
    # Fixed row positions (same as uniform keyboard)
    row_pitch = BASE_HEIGHT + PADDING
    row_tops = [0, -row_pitch, -2 * row_pitch]
    max_height_per_row = BASE_HEIGHT
    
    keys = []
    uniform_area = BASE_WIDTH * BASE_HEIGHT
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        total_width_budget = n * BASE_WIDTH
        total_height_budget = n * BASE_HEIGHT
        
        row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        # Apply scaling method
        if scale_method == 'linear':
            weights = {k: v for k, v in row_freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in row_freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
        else:
            weights = {k: v for k, v in row_freqs.items()}
        
        total_weight = sum(weights.values())
        normalized = {k: w / total_weight for k, w in weights.items()}
        
        # Calculate widths (redistributed, row total constant)
        widths = {k: normalized[k] * total_width_budget for k in row_keys}
        
        # Calculate heights (redistributed, then capped)
        heights = {k: normalized[k] * total_height_budget for k in row_keys}
        for letter in row_keys:
            heights[letter] = min(heights[letter], max_height_per_row)
        
        # Enforce minimums
        if enforce_minimum:
            # Width minimums
            deficit = 0
            for letter in row_keys:
                if widths[letter] < MIN_WIDTH:
                    deficit += MIN_WIDTH - widths[letter]
                    widths[letter] = MIN_WIDTH
            if deficit > 0:
                large_keys = [k for k in row_keys if widths[k] > MIN_WIDTH]
                large_total = sum(widths[k] for k in large_keys)
                for letter in large_keys:
                    widths[letter] -= deficit * (widths[letter] / large_total)
            
            # Height minimums
            for letter in row_keys:
                heights[letter] = max(heights[letter], MIN_HEIGHT)
        
        # Position keys
        x = row_idx * ROW_OFFSET
        row_top = row_tops[row_idx]
        
        for letter in row_keys:
            w = widths[letter]
            h = heights[letter]
            keys.append(Key(
                letter=letter,
                x=x,
                y=row_top - h,  # TOP aligned
                width=w,
                height=h,
                frequency=FREQUENCIES[letter],
                scale=(w * h) / uniform_area
            ))
            x += w + PADDING
    
    return keys


# =============================================================================
# Grid-Based Keyboard Generation
# =============================================================================

# Grid configuration
GRID_KEY_WIDTH = int(BASE_WIDTH)   # 7 columns per key
GRID_ROW_HEIGHT = int(BASE_HEIGHT) # 10 rows per keyboard row
GRID_ROW_OFFSETS = [0, int(ROW_OFFSET), int(2 * ROW_OFFSET)]  # [0, 4, 8] mm


def _smooth_grid_edges(grid_keys: Dict[str, 'GridKey'], 
                       key_info: Dict[str, dict],
                       iterations: int = 2,
                       window: int = 3) -> None:
    """
    Smooth the jagged edges of grid keys using a moving average filter.
    
    Modifies grid_keys in place.
    
    Args:
        grid_keys: Dictionary of letter -> GridKey
        key_info: Dictionary of letter -> {'row': int, 'cols': list}
        iterations: Number of smoothing passes
        window: Size of averaging window (must be odd)
    """
    if window % 2 == 0:
        window += 1  # Ensure odd window
    
    half_w = window // 2
    
    for _ in range(iterations):
        for letter, key in grid_keys.items():
            if not key.column_bounds:
                continue
            
            cols = sorted(key.column_bounds.keys())
            if len(cols) < window:
                continue
            
            # Get current top and bottom edges
            tops = [key.column_bounds[c][0] for c in cols]
            bottoms = [key.column_bounds[c][1] for c in cols]
            
            # Smooth internal columns (preserve edges)
            new_tops = tops.copy()
            new_bottoms = bottoms.copy()
            
            for i in range(half_w, len(cols) - half_w):
                # Average within window
                new_tops[i] = sum(tops[i-half_w:i+half_w+1]) / window
                new_bottoms[i] = sum(bottoms[i-half_w:i+half_w+1]) / window
            
            # Update column bounds
            for i, col in enumerate(cols):
                key.column_bounds[col] = (new_tops[i], new_bottoms[i])


def _enforce_grid_padding(grid_keys: Dict[str, 'GridKey'],
                          key_info: Dict[str, dict],
                          padding: float = PADDING) -> None:
    """
    Enforce minimum padding between vertically adjacent keys.
    
    After smoothing, keys might overlap. This function ensures minimum
    gaps by adjusting boundaries where keys meet.
    
    Modifies grid_keys in place.
    """
    # Find max column
    max_col = max(max(info['cols']) for info in key_info.values())
    
    # Process each column
    for col in range(max_col + 1):
        # Find all keys in this column, sorted by row
        keys_in_col = []
        for letter, info in key_info.items():
            if col in info['cols']:
                keys_in_col.append((info['row'], letter))
        
        if len(keys_in_col) < 2:
            continue
        
        keys_in_col.sort()  # Sort by row
        
        # Check each pair of adjacent keys
        for i in range(len(keys_in_col) - 1):
            _, letter1 = keys_in_col[i]
            _, letter2 = keys_in_col[i + 1]
            
            key1 = grid_keys[letter1]
            key2 = grid_keys[letter2]
            
            if col not in key1.column_bounds or col not in key2.column_bounds:
                continue
            
            top1, bottom1 = key1.column_bounds[col]
            top2, bottom2 = key2.column_bounds[col]
            
            # key1 is above key2, so bottom1 should be < top2 - padding
            gap = top2 - bottom1
            
            if gap < padding:
                # Need to create space
                shortage = padding - gap
                half_shortage = shortage / 2
                
                # Move key1 up and key2 down
                new_bottom1 = bottom1 - half_shortage
                new_top2 = top2 + half_shortage
                
                # Update bounds
                key1.column_bounds[col] = (top1, new_bottom1)
                key2.column_bounds[col] = (new_top2, bottom2)


def generate_grid_keyboard(scale_method: str = 'linear', 
                           enforce_minimum: bool = True,
                           min_height_fraction: float = 0.2,
                           smoothing: int = 2,
                           use_2d_base: bool = False) -> List[GridKey]:
    """
    Generate keyboard using grid-based column-by-column height redistribution.
    
    Keys compete for vertical space based on frequency. Higher frequency keys
    "steal" vertical space from lower frequency neighbors. Keys can have
    different heights at different columns, creating non-rectangular shapes.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log' - how to map frequency to weight
        enforce_minimum: Whether to enforce minimum key size per column
        min_height_fraction: Minimum height as fraction of slot (default 0.2 = 2mm)
        smoothing: Number of smoothing iterations for edges (0 = no smoothing)
        use_2d_base: If True, start with Log 2D widths instead of uniform
    
    Returns:
        List of GridKey objects
    """
    # Build key position info: which columns and row each key occupies
    key_info = {}  # letter -> {'row': int, 'cols': list of ints}
    
    if use_2d_base:
        # Use Log 2D widths as base (redistribute width within each row)
        for row_idx, row_keys in enumerate(ROWS):
            n = len(row_keys)
            total_key_width = n * GRID_KEY_WIDTH
            
            row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
            
            total_weight = sum(weights.values())
            widths = {k: (weights[k] / total_weight) * total_key_width for k in row_keys}
            
            # Enforce minimum width
            if enforce_minimum:
                min_w = MIN_WIDTH
                for letter in row_keys:
                    widths[letter] = max(widths[letter], min_w)
            
            x = float(GRID_ROW_OFFSETS[row_idx])
            for letter in row_keys:
                width = widths[letter]
                start_col = int(round(x))
                end_col = int(round(x + width))
                key_info[letter] = {
                    'row': row_idx,
                    'cols': list(range(start_col, max(start_col + 1, end_col)))
                }
                x += width
    else:
        # Uniform widths
        for row_idx, row_keys in enumerate(ROWS):
            x = GRID_ROW_OFFSETS[row_idx]
            for letter in row_keys:
                key_info[letter] = {
                    'row': row_idx,
                    'cols': list(range(x, x + GRID_KEY_WIDTH))
                }
                x += GRID_KEY_WIDTH  # No padding in grid (keys are adjacent)
    
    # Initialize GridKey objects
    grid_keys = {
        letter: GridKey(
            letter=letter,
            frequency=FREQUENCIES[letter],
            row=key_info[letter]['row'],
            column_bounds={}
        )
        for letter in FREQUENCIES
    }
    
    # Find the maximum column
    max_col = max(max(info['cols']) for info in key_info.values())
    
    # Total height of keyboard in grid units
    total_height = len(ROWS) * GRID_ROW_HEIGHT  # 30mm
    
    # Process each column independently
    for col in range(max_col + 1):
        # Find all keys present in this column
        keys_here = []  # [(row_idx, letter), ...]
        for letter, info in key_info.items():
            if col in info['cols']:
                keys_here.append((info['row'], letter))
        
        if not keys_here:
            continue
        
        # Sort by row (top to bottom: row 0, 1, 2)
        keys_here.sort()
        
        # Determine which row slots are occupied
        occupied_rows = [row for row, _ in keys_here]
        first_row = min(occupied_rows)
        last_row = max(occupied_rows)
        
        # Height budget = vertical extent from first to last occupied row
        # Each row slot is GRID_ROW_HEIGHT (10mm) plus PADDING between rows
        row_pitch = GRID_ROW_HEIGHT + PADDING
        # Total span from first row top to last row bottom
        height_budget = (last_row - first_row) * row_pitch + GRID_ROW_HEIGHT
        # Subtract padding between keys within this column
        num_gaps = len(keys_here) - 1
        height_budget -= num_gaps * PADDING
        
        # Calculate weights based on frequency
        weights = []
        for row, letter in keys_here:
            freq = FREQUENCIES[letter]
            if scale_method == 'sqrt':
                weights.append(np.sqrt(freq))
            elif scale_method == 'log':
                weights.append(np.log1p(freq))
            else:
                weights.append(freq)
        
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1  # Avoid division by zero
        
        # Calculate raw heights based on frequency
        raw_heights = [w / total_weight * height_budget for w in weights]
        
        # Enforce minimum height
        if enforce_minimum:
            min_h = min_height_fraction * GRID_ROW_HEIGHT
            heights = []
            deficit = 0
            
            for h in raw_heights:
                if h < min_h:
                    deficit += min_h - h
                    heights.append(min_h)
                else:
                    heights.append(h)
            
            # Redistribute deficit from larger keys
            if deficit > 0:
                large_indices = [i for i, h in enumerate(heights) if h > min_h]
                large_total = sum(heights[i] for i in large_indices)
                if large_total > 0:
                    for i in large_indices:
                        heights[i] -= deficit * (heights[i] / large_total)
        else:
            heights = raw_heights
        
        # Assign top/bottom positions (stack from top of first row's slot)
        # Y increases downward in our coordinate system
        current_y = first_row * row_pitch  # Top of first occupied row
        
        for i, ((row, letter), h) in enumerate(zip(keys_here, heights)):
            top_y = current_y
            bottom_y = current_y + h
            grid_keys[letter].column_bounds[col] = (top_y, bottom_y)
            current_y = bottom_y
            # Add padding gap before next key (if not last)
            if i < len(keys_here) - 1:
                current_y += PADDING
    
    # Apply edge smoothing
    if smoothing > 0:
        _smooth_grid_edges(grid_keys, key_info, iterations=smoothing)
    
    # Enforce padding after smoothing (smoothing can cause overlaps)
    _enforce_grid_padding(grid_keys, key_info, padding=PADDING)
    
    return list(grid_keys.values())


def generate_grid_keyboard_from_2d(scale_method: str = 'log',
                                    enforce_minimum: bool = True,
                                    min_height_fraction: float = 0.2,
                                    smoothing: int = 4) -> List[GridKey]:
    """
    Generate grid keyboard starting from 2D (width-redistributed) layout.
    
    Instead of uniform key widths, uses frequency-based widths from the 2D
    approach, then applies grid-based height redistribution on top.
    
    This combines:
    - Horizontal: Width redistribution within each row (from 2D approach)
    - Vertical: Column-by-column height competition (from grid approach)
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log' - how to map frequency to weight
        enforce_minimum: Whether to enforce minimum key size
        min_height_fraction: Minimum height as fraction of slot
        smoothing: Number of smoothing iterations for edges
    
    Returns:
        List of GridKey objects
    """
    # Step 1: Calculate widths for each key (same as generate_keyboard)
    key_widths = {}  # letter -> width in mm
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        total_key_width = n * BASE_WIDTH
        
        row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        if scale_method == 'linear':
            weights = {k: v for k, v in row_freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in row_freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
        else:
            weights = {k: v for k, v in row_freqs.items()}
        
        total_weight = sum(weights.values())
        widths = {k: (weights[k] / total_weight) * total_key_width for k in row_keys}
        
        # Enforce minimum width
        if enforce_minimum:
            deficit = 0
            for letter in row_keys:
                if widths[letter] < MIN_WIDTH:
                    deficit += MIN_WIDTH - widths[letter]
                    widths[letter] = MIN_WIDTH
            if deficit > 0:
                large_keys = [k for k in row_keys if widths[k] > MIN_WIDTH]
                large_total = sum(widths[k] for k in large_keys)
                for letter in large_keys:
                    widths[letter] -= deficit * (widths[letter] / large_total)
        
        key_widths.update(widths)
    
    # Step 2: Build key position info with variable widths
    # Convert widths to column ranges (1mm per column)
    key_info = {}  # letter -> {'row': int, 'cols': list of ints}
    
    for row_idx, row_keys in enumerate(ROWS):
        x = float(GRID_ROW_OFFSETS[row_idx])
        for letter in row_keys:
            width = key_widths[letter]
            start_col = int(round(x))
            end_col = int(round(x + width))
            key_info[letter] = {
                'row': row_idx,
                'cols': list(range(start_col, end_col))
            }
            x += width  # No padding between keys in grid
    
    # Step 3: Initialize GridKey objects
    grid_keys = {
        letter: GridKey(
            letter=letter,
            frequency=FREQUENCIES[letter],
            row=key_info[letter]['row'],
            column_bounds={}
        )
        for letter in FREQUENCIES
    }
    
    # Step 4: Find max column and process each column for height redistribution
    max_col = max(max(info['cols']) for info in key_info.values() if info['cols'])
    row_pitch = GRID_ROW_HEIGHT + PADDING
    
    for col in range(max_col + 1):
        keys_here = []
        for letter, info in key_info.items():
            if col in info['cols']:
                keys_here.append((info['row'], letter))
        
        if not keys_here:
            continue
        
        keys_here.sort()  # Sort by row
        
        occupied_rows = [row for row, _ in keys_here]
        first_row = min(occupied_rows)
        last_row = max(occupied_rows)
        
        # Height budget
        height_budget = (last_row - first_row) * row_pitch + GRID_ROW_HEIGHT
        num_gaps = len(keys_here) - 1
        height_budget -= num_gaps * PADDING
        
        # Calculate weights
        weights = []
        for row, letter in keys_here:
            freq = FREQUENCIES[letter]
            if scale_method == 'sqrt':
                weights.append(np.sqrt(freq))
            elif scale_method == 'log':
                weights.append(np.log1p(freq))
            else:
                weights.append(freq)
        
        total_weight = sum(weights) or 1
        raw_heights = [w / total_weight * height_budget for w in weights]
        
        # Enforce minimum height
        if enforce_minimum:
            min_h = min_height_fraction * GRID_ROW_HEIGHT
            heights = []
            deficit = 0
            for h in raw_heights:
                if h < min_h:
                    deficit += min_h - h
                    heights.append(min_h)
                else:
                    heights.append(h)
            if deficit > 0:
                large_indices = [i for i, h in enumerate(heights) if h > min_h]
                large_total = sum(heights[i] for i in large_indices)
                if large_total > 0:
                    for i in large_indices:
                        heights[i] -= deficit * (heights[i] / large_total)
        else:
            heights = raw_heights
        
        # Position keys
        current_y = first_row * row_pitch
        for i, ((row, letter), h) in enumerate(zip(keys_here, heights)):
            top_y = current_y
            bottom_y = current_y + h
            grid_keys[letter].column_bounds[col] = (top_y, bottom_y)
            current_y = bottom_y
            if i < len(keys_here) - 1:
                current_y += PADDING
    
    # Step 5: Apply smoothing and enforce padding
    if smoothing > 0:
        _smooth_grid_edges(grid_keys, key_info, iterations=smoothing)
    
    _enforce_grid_padding(grid_keys, key_info, padding=PADDING)
    
    return list(grid_keys.values())


# =============================================================================
# Graph-Based Neighbor-Aware Keyboard Generation
# =============================================================================

def _build_neighbor_graph() -> Dict[str, Dict[str, str]]:
    """
    Build a graph of neighboring keys based on QWERTY layout.
    
    For each key, returns which keys are adjacent in each direction:
    - 'left', 'right': Same row neighbors
    - 'above_left', 'above', 'above_right': Row above (accounting for offset)
    - 'below_left', 'below', 'below_right': Row below (accounting for offset)
    
    Returns:
        Dict mapping letter -> Dict of direction -> neighbor letter (or None)
    """
    neighbors = {}
    
    # Build position map: letter -> (row, position_in_row)
    positions = {}
    for row_idx, row_keys in enumerate(ROWS):
        for pos, letter in enumerate(row_keys):
            positions[letter] = (row_idx, pos)
    
    # Row offset in "half-key" units (row 1 is offset ~0.5 keys right from row 0)
    # Row 0: starts at x=0
    # Row 1: starts at x=4mm (about 0.57 of a key width)
    # Row 2: starts at x=8mm (about 1.14 of a key width)
    
    for letter in FREQUENCIES:
        row_idx, pos = positions[letter]
        n = {}
        
        # Same row neighbors
        if pos > 0:
            n['left'] = ROWS[row_idx][pos - 1]
        if pos < len(ROWS[row_idx]) - 1:
            n['right'] = ROWS[row_idx][pos + 1]
        
        # Calculate x position (center of key) in mm
        x_center = GRID_ROW_OFFSETS[row_idx] + pos * GRID_KEY_WIDTH + GRID_KEY_WIDTH / 2
        
        # Row above
        if row_idx > 0:
            above_row = ROWS[row_idx - 1]
            above_offset = GRID_ROW_OFFSETS[row_idx - 1]
            
            for above_pos, above_letter in enumerate(above_row):
                above_x = above_offset + above_pos * GRID_KEY_WIDTH + GRID_KEY_WIDTH / 2
                dx = above_x - x_center
                
                if abs(dx) < GRID_KEY_WIDTH * 0.7:
                    # Directly above (overlapping)
                    if 'above' not in n:
                        n['above'] = above_letter
                elif -GRID_KEY_WIDTH * 1.5 < dx < -GRID_KEY_WIDTH * 0.3:
                    n['above_left'] = above_letter
                elif GRID_KEY_WIDTH * 0.3 < dx < GRID_KEY_WIDTH * 1.5:
                    n['above_right'] = above_letter
        
        # Row below
        if row_idx < len(ROWS) - 1:
            below_row = ROWS[row_idx + 1]
            below_offset = GRID_ROW_OFFSETS[row_idx + 1]
            
            for below_pos, below_letter in enumerate(below_row):
                below_x = below_offset + below_pos * GRID_KEY_WIDTH + GRID_KEY_WIDTH / 2
                dx = below_x - x_center
                
                if abs(dx) < GRID_KEY_WIDTH * 0.7:
                    if 'below' not in n:
                        n['below'] = below_letter
                elif -GRID_KEY_WIDTH * 1.5 < dx < -GRID_KEY_WIDTH * 0.3:
                    n['below_left'] = below_letter
                elif GRID_KEY_WIDTH * 0.3 < dx < GRID_KEY_WIDTH * 1.5:
                    n['below_right'] = below_letter
        
        neighbors[letter] = n
    
    return neighbors


def _negotiate_boundary(freq1: float, freq2: float, 
                        scale_method: str = 'log',
                        neutral: float = 0.5) -> float:
    """
    Negotiate where the boundary should be between two keys.
    
    Returns a value 0-1 where:
    - 0.5 = even split (neutral)
    - <0.5 = boundary moves toward key1 (key2 wins more space)
    - >0.5 = boundary moves toward key2 (key1 wins more space)
    
    Args:
        freq1: Frequency of first key
        freq2: Frequency of second key
        scale_method: How to weight frequencies ('linear', 'sqrt', 'log')
        neutral: The neutral split point (default 0.5)
    """
    if scale_method == 'sqrt':
        w1, w2 = np.sqrt(freq1), np.sqrt(freq2)
    elif scale_method == 'log':
        w1, w2 = np.log1p(freq1), np.log1p(freq2)
    else:
        w1, w2 = freq1, freq2
    
    total = w1 + w2
    if total == 0:
        return neutral
    
    return w1 / total


def _check_overlap(b1: Dict, b2: Dict, padding: float = PADDING) -> Tuple[bool, float, float]:
    """
    Check if two key bounds overlap (including padding requirement).
    
    Returns:
        (overlaps, x_overlap, y_overlap) - overlap amounts (negative = gap exists)
    """
    # Calculate overlap in each dimension
    x_overlap = min(b1['right'], b2['right']) - max(b1['left'], b2['left'])
    y_overlap = min(b1['bottom'], b2['bottom']) - max(b1['top'], b2['top'])
    
    # They overlap if both dimensions have positive overlap
    # But we also need padding, so treat overlap as: overlap > -padding
    overlaps = (x_overlap > -padding) and (y_overlap > -padding)
    
    return overlaps, x_overlap, y_overlap


def _resolve_overlaps(key_bounds: Dict[str, Dict], 
                      neighbors: Dict[str, Dict[str, str]],
                      padding: float = PADDING) -> Dict[str, Dict]:
    """
    Detect and resolve any overlaps between keys.
    
    For each pair of overlapping keys, push them apart to maintain padding.
    """
    letters = list(key_bounds.keys())
    resolved = {k: dict(v) for k, v in key_bounds.items()}
    
    # Check all pairs of keys
    for i, letter1 in enumerate(letters):
        for letter2 in letters[i+1:]:
            b1 = resolved[letter1]
            b2 = resolved[letter2]
            
            overlaps, x_overlap, y_overlap = _check_overlap(b1, b2, padding)
            
            if not overlaps:
                continue
            
            # Determine relationship based on rows
            row1, row2 = b1['row'], b2['row']
            
            if row1 == row2:
                # Same row - resolve horizontally
                if x_overlap > -padding:
                    # Need to separate by (x_overlap + padding)
                    separation_needed = x_overlap + padding
                    
                    # Determine which is left, which is right
                    if b1['left'] < b2['left']:
                        left_key, right_key = letter1, letter2
                    else:
                        left_key, right_key = letter2, letter1
                    
                    # Split the separation proportionally by frequency
                    freq_left = FREQUENCIES[left_key]
                    freq_right = FREQUENCIES[right_key]
                    total_freq = freq_left + freq_right
                    
                    # Higher freq key gives up less
                    left_share = (freq_right / total_freq) if total_freq > 0 else 0.5
                    
                    resolved[left_key]['right'] -= separation_needed * left_share
                    resolved[right_key]['left'] += separation_needed * (1 - left_share)
            
            else:
                # Different rows - resolve vertically if they overlap horizontally
                if x_overlap > 0 and y_overlap > -padding:
                    separation_needed = y_overlap + padding
                    
                    # Determine which is above, which is below
                    if b1['row'] < b2['row']:
                        above_key, below_key = letter1, letter2
                    else:
                        above_key, below_key = letter2, letter1
                    
                    freq_above = FREQUENCIES[above_key]
                    freq_below = FREQUENCIES[below_key]
                    total_freq = freq_above + freq_below
                    
                    above_share = (freq_below / total_freq) if total_freq > 0 else 0.5
                    
                    resolved[above_key]['bottom'] -= separation_needed * above_share
                    resolved[below_key]['top'] += separation_needed * (1 - above_share)
    
    return resolved


def generate_cell_keyboard(scale_method: str = 'log',
                           iterations: int = 10,
                           expansion_rate: float = 0.3,
                           use_2d_base: bool = True) -> List[GridKey]:
    """
    Generate keyboard using cell-based competition.
    
    Each 1mm x 1mm cell is assigned to a key. High-frequency keys
    expand into neighboring low-frequency keys' cells over iterations.
    Creates truly organic, non-rectangular shapes with variable heights.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
        iterations: Number of expansion iterations
        expansion_rate: How aggressively high-freq keys expand (0-1)
        use_2d_base: If True, start with Log 2D widths (default True)
    
    Returns:
        List of GridKey objects
    """
    # Use continuous grid without padding gaps - padding applied at render
    total_rows = len(ROWS)
    row_height = GRID_ROW_HEIGHT
    
    # Grid dimensions - no gaps between rows in the cell grid
    grid_width = 80  # mm
    grid_height = int(total_rows * row_height)  # 30mm for 3 rows
    
    # Initialize cell ownership: cell_owner[y][x] = letter or None
    cell_owner = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Calculate key weights based on frequency
    def get_weight(letter):
        freq = FREQUENCIES[letter]
        if scale_method == 'sqrt':
            return np.sqrt(freq)
        elif scale_method == 'log':
            return np.log1p(freq)
        return freq
    
    weights = {letter: get_weight(letter) for letter in FREQUENCIES}
    max_weight = max(weights.values())
    norm_weights = {k: v / max_weight for k, v in weights.items()}
    
    # Calculate initial widths - use Log 2D base if requested
    key_widths = {}
    if use_2d_base:
        for row_idx, row_keys in enumerate(ROWS):
            n = len(row_keys)
            total_key_width = n * GRID_KEY_WIDTH
            
            row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
            row_weights = {k: np.log1p(v) for k, v in row_freqs.items()}
            
            total_weight = sum(row_weights.values())
            for letter in row_keys:
                key_widths[letter] = (row_weights[letter] / total_weight) * total_key_width
                key_widths[letter] = max(key_widths[letter], MIN_WIDTH)
    else:
        for letter in FREQUENCIES:
            key_widths[letter] = GRID_KEY_WIDTH
    
    # Initial assignment: each key gets a region based on calculated widths
    key_regions = {}
    
    for row_idx, row_keys in enumerate(ROWS):
        x = float(GRID_ROW_OFFSETS[row_idx])
        y_top = int(row_idx * row_height)
        y_bottom = int(y_top + row_height)
        
        for letter in row_keys:
            width = key_widths[letter]
            x_start = int(x)
            x_end = int(x + width)
            
            key_regions[letter] = {
                'row': row_idx,
                'x_start': x_start,
                'x_end': x_end,
                'y_start': y_top,
                'y_end': y_bottom
            }
            
            # Assign cells to this key (fill entire row height)
            for cy in range(y_top, min(y_bottom, grid_height)):
                for cx in range(x_start, min(x_end, grid_width)):
                    cell_owner[cy][cx] = letter
            
            x += width  # Use calculated width
    
    # Iteratively expand high-frequency keys into low-frequency neighbors
    for iteration in range(iterations):
        # Create a copy for this iteration
        new_owner = [row[:] for row in cell_owner]
        
        # Decay factor - less aggressive in later iterations
        decay = 1.0 - (iteration / iterations) * 0.5
        
        for y in range(1, grid_height - 1):
            for x in range(1, grid_width - 1):
                current = cell_owner[y][x]
                if current is None:
                    continue
                
                current_weight = norm_weights[current]
                
                # Check all 8 neighbors
                neighbors = [
                    (y-1, x-1), (y-1, x), (y-1, x+1),
                    (y, x-1),            (y, x+1),
                    (y+1, x-1), (y+1, x), (y+1, x+1)
                ]
                
                # Find the strongest neighbor
                strongest_neighbor = None
                strongest_weight = current_weight
                
                for ny, nx in neighbors:
                    if 0 <= ny < grid_height and 0 <= nx < grid_width:
                        neighbor = cell_owner[ny][nx]
                        if neighbor is not None and neighbor != current:
                            neighbor_weight = norm_weights[neighbor]
                            if neighbor_weight > strongest_weight:
                                strongest_weight = neighbor_weight
                                strongest_neighbor = neighbor
                
                # If a neighbor is significantly stronger, it might claim this cell
                if strongest_neighbor is not None:
                    weight_ratio = strongest_weight / (current_weight + 0.01)
                    
                    # Probability of takeover based on weight ratio
                    takeover_prob = (weight_ratio - 1) * expansion_rate * decay
                    
                    # Count how many neighbors are the strong key (more neighbors = more likely)
                    strong_neighbor_count = sum(
                        1 for ny, nx in neighbors
                        if 0 <= ny < grid_height and 0 <= nx < grid_width
                        and cell_owner[ny][nx] == strongest_neighbor
                    )
                    
                    # Need at least 2 neighboring cells of the strong key
                    if strong_neighbor_count >= 2 and takeover_prob > 0.1:
                        new_owner[y][x] = strongest_neighbor
        
        cell_owner = new_owner
    
    # Ensure minimum connectivity - each key must have contiguous cells
    # (Skip for now, can add if needed)
    
    # Convert cell ownership to GridKey column_bounds
    # Need to transform y-coordinates to include padding between rows for visualization
    row_pitch = GRID_ROW_HEIGHT + PADDING
    
    def transform_y(cell_y):
        """Transform cell grid y to visualization y (with padding gaps)."""
        # Which row is this cell in?
        row = cell_y // row_height
        # Position within row
        y_in_row = cell_y % row_height
        # Add padding for each row boundary crossed
        return row * row_pitch + y_in_row
    
    grid_keys = {}
    for letter in FREQUENCIES:
        grid_keys[letter] = GridKey(
            letter=letter,
            frequency=FREQUENCIES[letter],
            row=key_regions[letter]['row'],
            column_bounds={}
        )
    
    # For each column (x), find the y-range owned by each key
    for x in range(grid_width):
        # Group y values by key
        column_keys = {}  # letter -> list of y values
        
        for y in range(grid_height):
            owner = cell_owner[y][x]
            if owner is not None:
                if owner not in column_keys:
                    column_keys[owner] = []
                column_keys[owner].append(y)
        
        # For each key, store the transformed y range
        for letter, y_values in column_keys.items():
            if y_values:
                # Transform to visualization coordinates
                y_min = transform_y(min(y_values))
                y_max = transform_y(max(y_values) + 1)
                
                # Only add if this is a significant presence in this column
                if len(y_values) >= 2:
                    if x not in grid_keys[letter].column_bounds:
                        grid_keys[letter].column_bounds[x] = (y_min, y_max)
                    else:
                        # Merge with existing
                        old_min, old_max = grid_keys[letter].column_bounds[x]
                        grid_keys[letter].column_bounds[x] = (min(old_min, y_min), max(old_max, y_max))
    
    # Filter out keys with no cells
    result = [key for key in grid_keys.values() if key.column_bounds]
    
    return result


def generate_tiled_keyboard(scale_method: str = 'log',
                            smoothing: int = 3,
                            min_size_fraction: float = 0.25) -> List[GridKey]:
    """
    Generate keyboard with true tiling - keys have jagged edges that interlock.
    
    Each column's vertical boundary is negotiated independently based on the
    specific keys above/below at that x position. Creates puzzle-piece-like
    interlocking shapes.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
        smoothing: Number of smoothing passes
        min_size_fraction: Minimum height as fraction of row height
    
    Returns:
        List of GridKey objects
    """
    row_pitch = GRID_ROW_HEIGHT + PADDING
    
    # Step 1: Calculate widths for each key (redistributed within row)
    key_widths = {}
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        total_key_width = n * GRID_KEY_WIDTH
        
        row_freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        if scale_method == 'linear':
            weights = {k: v for k, v in row_freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in row_freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in row_freqs.items()}
        else:
            weights = {k: v for k, v in row_freqs.items()}
        
        total_weight = sum(weights.values())
        widths = {k: (weights[k] / total_weight) * total_key_width for k in row_keys}
        
        # Enforce minimum width
        min_w = GRID_KEY_WIDTH * min_size_fraction
        for letter in row_keys:
            widths[letter] = max(widths[letter], min_w)
        
        key_widths.update(widths)
    
    # Step 2: Build key position info with variable widths
    key_info = {}  # letter -> {'row': int, 'x_start': float, 'x_end': float}
    
    for row_idx, row_keys in enumerate(ROWS):
        x = float(GRID_ROW_OFFSETS[row_idx])
        for letter in row_keys:
            width = key_widths[letter]
            key_info[letter] = {
                'row': row_idx,
                'x_start': x,
                'x_end': x + width
            }
            x += width
    
    # Step 3: For each column, determine which keys are present and negotiate heights
    max_col = int(max(info['x_end'] for info in key_info.values())) + 1
    
    # Initialize column bounds for each key
    grid_keys = {
        letter: GridKey(
            letter=letter,
            frequency=FREQUENCIES[letter],
            row=key_info[letter]['row'],
            column_bounds={}
        )
        for letter in FREQUENCIES
    }
    
    # Process each column
    for col in range(max_col):
        col_center = col + 0.5
        
        # Find keys that cover this column
        keys_in_col = []
        for letter, info in key_info.items():
            if info['x_start'] <= col_center < info['x_end']:
                keys_in_col.append((info['row'], letter))
        
        if not keys_in_col:
            continue
        
        keys_in_col.sort()  # Sort by row (top to bottom)
        
        # Get row range
        first_row = keys_in_col[0][0]
        last_row = keys_in_col[-1][0]
        
        # Total height available for these keys
        total_height = (last_row - first_row) * row_pitch + GRID_ROW_HEIGHT
        
        # Subtract padding between keys
        num_gaps = len(keys_in_col) - 1
        available_height = total_height - (num_gaps * PADDING)
        
        # Calculate weights for height distribution
        weights = []
        for row, letter in keys_in_col:
            freq = FREQUENCIES[letter]
            if scale_method == 'sqrt':
                weights.append(np.sqrt(freq))
            elif scale_method == 'log':
                weights.append(np.log1p(freq))
            else:
                weights.append(freq)
        
        total_weight = sum(weights) or 1
        
        # Calculate heights
        heights = [(w / total_weight) * available_height for w in weights]
        
        # Enforce minimum height
        min_h = GRID_ROW_HEIGHT * min_size_fraction
        for i in range(len(heights)):
            heights[i] = max(heights[i], min_h)
        
        # Normalize to fit available space
        height_sum = sum(heights)
        if height_sum > available_height:
            scale_factor = available_height / height_sum
            heights = [h * scale_factor for h in heights]
        
        # Position keys vertically with padding
        current_y = first_row * row_pitch
        
        for i, (row, letter) in enumerate(keys_in_col):
            top_y = current_y
            bottom_y = current_y + heights[i]
            grid_keys[letter].column_bounds[col] = (top_y, bottom_y)
            current_y = bottom_y + PADDING
    
    # Step 4: Apply smoothing to reduce jaggedness while keeping tiling
    if smoothing > 0:
        _smooth_grid_edges(grid_keys, 
                          {k: {'row': v['row'], 'cols': list(v.column_bounds.keys())} 
                           for k, v in grid_keys.items()},
                          iterations=smoothing, window=3)
    
    # Step 5: Ensure no overlaps after smoothing
    _enforce_grid_padding(grid_keys,
                         {k: {'row': v['row'], 'cols': list(v.column_bounds.keys())} 
                          for k, v in grid_keys.items()},
                         padding=PADDING)
    
    return list(grid_keys.values())


def generate_graph_keyboard(scale_method: str = 'log',
                            iterations: int = 15,
                            min_size_fraction: float = 0.3) -> List[GridKey]:
    """
    Generate keyboard using graph-based neighbor negotiation with overlap detection.
    
    Each key's boundaries are determined by negotiating with its neighbors
    based on relative frequency. Overlaps are detected and resolved after
    each iteration.
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
        iterations: Number of negotiation iterations
        min_size_fraction: Minimum size as fraction of uniform (prevents tiny keys)
    
    Returns:
        List of GridKey objects
    """
    neighbors = _build_neighbor_graph()
    row_pitch = GRID_ROW_HEIGHT + PADDING
    
    # Initialize each key with uniform bounds (with padding between rows)
    key_bounds = {}
    
    for row_idx, row_keys in enumerate(ROWS):
        x = float(GRID_ROW_OFFSETS[row_idx])
        y_top = row_idx * row_pitch
        y_bottom = y_top + GRID_ROW_HEIGHT
        
        for letter in row_keys:
            key_bounds[letter] = {
                'left': x + PADDING/2,  # Add padding on edges
                'right': x + GRID_KEY_WIDTH - PADDING/2,
                'top': y_top,
                'bottom': y_bottom,
                'row': row_idx
            }
            x += GRID_KEY_WIDTH
    
    # Store original positions for reference
    original_bounds = {k: dict(v) for k, v in key_bounds.items()}
    
    # Iteratively adjust boundaries based on neighbor negotiations
    for iteration in range(iterations):
        # Damping factor decreases over iterations for stability
        damping = 1.0 - (iteration / iterations) * 0.5
        
        new_bounds = {k: dict(v) for k, v in key_bounds.items()}
        
        # Process each key's negotiations with neighbors
        for letter, n in neighbors.items():
            freq = FREQUENCIES[letter]
            bounds = key_bounds[letter]
            orig = original_bounds[letter]
            
            # Negotiate with same-row neighbors (horizontal boundaries)
            if 'left' in n:
                left_neighbor = n['left']
                left_freq = FREQUENCIES[left_neighbor]
                left_bound = key_bounds[left_neighbor]
                
                # Share based on relative frequency
                share = _negotiate_boundary(freq, left_freq, scale_method)
                
                # Calculate new boundary position
                total_space = left_bound['right'] - bounds['left'] + GRID_KEY_WIDTH
                my_share = share * (GRID_KEY_WIDTH - PADDING)
                
                # Shift from original midpoint
                orig_boundary = (orig['left'] + original_bounds[left_neighbor]['right']) / 2
                shift = (share - 0.5) * GRID_KEY_WIDTH * 0.4 * damping
                
                new_boundary = orig_boundary + shift
                new_bounds[letter]['left'] = new_boundary + PADDING/2
                new_bounds[left_neighbor]['right'] = new_boundary - PADDING/2
            
            if 'right' in n:
                right_neighbor = n['right']
                right_freq = FREQUENCIES[right_neighbor]
                right_bound = key_bounds[right_neighbor]
                
                share = _negotiate_boundary(freq, right_freq, scale_method)
                
                orig_boundary = (orig['right'] + original_bounds[right_neighbor]['left']) / 2
                shift = (share - 0.5) * GRID_KEY_WIDTH * 0.4 * damping
                
                new_boundary = orig_boundary + shift
                new_bounds[letter]['right'] = new_boundary - PADDING/2
                new_bounds[right_neighbor]['left'] = new_boundary + PADDING/2
            
            # Negotiate with keys in row below (vertical boundaries)
            for direction in ['below', 'below_left', 'below_right']:
                if direction in n:
                    below_neighbor = n[direction]
                    below_freq = FREQUENCIES[below_neighbor]
                    below_bound = key_bounds[below_neighbor]
                    below_orig = original_bounds[below_neighbor]
                    
                    share = _negotiate_boundary(freq, below_freq, scale_method)
                    
                    # Original boundary between rows
                    orig_boundary = (orig['bottom'] + below_orig['top']) / 2
                    shift = (share - 0.5) * GRID_ROW_HEIGHT * 0.3 * damping
                    
                    new_boundary = orig_boundary + shift
                    new_bounds[letter]['bottom'] = new_boundary - PADDING/2
                    new_bounds[below_neighbor]['top'] = new_boundary + PADDING/2
        
        # Enforce minimum sizes
        min_w = GRID_KEY_WIDTH * min_size_fraction
        min_h = GRID_ROW_HEIGHT * min_size_fraction
        
        for letter in FREQUENCIES:
            b = new_bounds[letter]
            width = b['right'] - b['left']
            height = b['bottom'] - b['top']
            
            if width < min_w:
                center = (b['left'] + b['right']) / 2
                b['left'] = center - min_w / 2
                b['right'] = center + min_w / 2
            
            if height < min_h:
                center = (b['top'] + b['bottom']) / 2
                b['top'] = center - min_h / 2
                b['bottom'] = center + min_h / 2
        
        # Resolve any overlaps
        new_bounds = _resolve_overlaps(new_bounds, neighbors, PADDING)
        
        key_bounds = new_bounds
    
    # Final overlap check and resolution
    for _ in range(5):
        key_bounds = _resolve_overlaps(key_bounds, neighbors, PADDING)
    
    # Convert bounds to GridKey objects
    grid_keys = []
    for letter in FREQUENCIES:
        b = key_bounds[letter]
        
        # Create column bounds from left/right and top/bottom
        left_col = int(np.floor(b['left']))
        right_col = int(np.ceil(b['right']))
        
        column_bounds = {}
        for col in range(left_col, right_col):
            # Clip column bounds to actual key bounds
            col_left = max(col, b['left'])
            col_right = min(col + 1, b['right'])
            if col_right > col_left:
                column_bounds[col] = (b['top'], b['bottom'])
        
        if column_bounds:  # Only add if key has at least one column
            grid_keys.append(GridKey(
                letter=letter,
                frequency=FREQUENCIES[letter],
                row=b['row'],
                column_bounds=column_bounds
            ))
    
    return grid_keys


# =============================================================================
# Unit Grid Keyboard System
# =============================================================================

# Unit grid constants
SLOT_WIDTH = 9    # units (7mm key + 1mm padding each side)
SLOT_HEIGHT = 12  # units (10mm key + 1mm padding each side)
ACTIVE_WIDTH = 7  # units (actual key width, no padding)
ACTIVE_HEIGHT = 10  # units (actual key height, no padding)
PAD = 1           # units of padding on each side
ROW_GAP = 1       # units of gap between rows

# Row offsets in units (matching mm offsets)
ROW_OFFSETS_UNITS = [0, 4, 8]


@dataclass
class UnitGridKeyboard:
    """
    Keyboard represented as a grid of 1mm × 1mm unit cells.
    Each cell is owned by exactly one key.
    """
    grid: List[List[str]]  # grid[y][x] = letter (owner of cell)
    width: int
    height: int
    key_bounds: Dict[str, Tuple[int, int, int, int]]  # letter -> (x, y, w, h)
    
    def get_key_cells(self, letter: str) -> List[Tuple[int, int]]:
        """Get all (x, y) cells owned by a key."""
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == letter:
                    cells.append((x, y))
        return cells
    
    def get_key_area(self, letter: str) -> int:
        """Get total area (in units) owned by a key."""
        return len(self.get_key_cells(letter))
    
    def get_active_bounds(self, letter: str) -> Tuple[int, int, int, int]:
        """
        Get active (non-padding) bounds for a key.
        Returns (x, y, width, height) of the inner rectangle after removing 1-unit padding.
        """
        cells = self.get_key_cells(letter)
        if not cells:
            return (0, 0, 0, 0)
        
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        
        # Full bounds
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Active bounds (1-unit padding inward)
        active_x = x_min + PAD
        active_y = y_min + PAD
        active_w = max(0, (x_max - x_min + 1) - 2 * PAD)
        active_h = max(0, (y_max - y_min + 1) - 2 * PAD)
        
        return (active_x, active_y, active_w, active_h)
    

def generate_unit_grid_keyboard() -> UnitGridKeyboard:
    """
    Generate keyboard as a grid of 1mm × 1mm unit cells.
    Each key gets a 9×12 slot (7×10 key + 1-unit padding on all sides).
    
    Returns:
        UnitGridKeyboard with all cells assigned to keys (uniform sizing)
    """
    # Calculate grid dimensions
    row_widths = [
        ROW_OFFSETS_UNITS[i] + len(ROWS[i]) * SLOT_WIDTH
        for i in range(len(ROWS))
    ]
    grid_width = max(row_widths)
    grid_height = len(ROWS) * SLOT_HEIGHT
    
    # Create grid (None = unassigned)
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    key_bounds = {}
    
    # Assign slots to keys (with gaps between rows)
    for row_idx, row_keys in enumerate(ROWS):
        x = ROW_OFFSETS_UNITS[row_idx]
        y = row_idx * (SLOT_HEIGHT + ROW_GAP)
        
        for letter in row_keys:
            key_bounds[letter] = (x, y, SLOT_WIDTH, SLOT_HEIGHT)
            
            # Assign all cells in slot to this key
            for dy in range(SLOT_HEIGHT):
                for dx in range(SLOT_WIDTH):
                    if y + dy < grid_height and x + dx < grid_width:
                        grid[y + dy][x + dx] = letter
            
            x += SLOT_WIDTH
    
    return UnitGridKeyboard(
        grid=grid,
        width=grid_width,
        height=grid_height,
        key_bounds=key_bounds
    )


def generate_unit_grid_log2d(scale_method: str = 'log') -> UnitGridKeyboard:
    """
    Generate unit grid with Log 2D sizing matching Round 3's generate_keyboard_2d().
    
    Replicates the exact sizing algorithm from Round 3:
    - Width: Redistributed within each row (row total constant)
    - Height: Redistributed within each row, capped at slot height
    - Minimum sizes enforced
    - Row gap: 1 unit between rows for visual separation
    
    Args:
        scale_method: 'linear', 'sqrt', or 'log'
    
    Returns:
        UnitGridKeyboard with frequency-sized keys matching Round 3
    """
    # Grid dimensions based on uniform slots (with gaps between rows)
    row_widths_max = [
        ROW_OFFSETS_UNITS[i] + len(ROWS[i]) * SLOT_WIDTH
        for i in range(len(ROWS))
    ]
    grid_width = max(row_widths_max)
    # Height includes gaps between rows
    grid_height = len(ROWS) * SLOT_HEIGHT + (len(ROWS) - 1) * ROW_GAP
    
    # Create grid (None = unassigned)
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    key_bounds = {}
    
    # Minimum sizes (matching MIN_WIDTH=3, MIN_HEIGHT=5 from Round 3)
    min_w = 3
    min_h = 5
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        # Match Round 3: total key width = n * BASE_WIDTH (7mm)
        total_key_width = n * ACTIVE_WIDTH  # 7 units per key base
        total_height_budget = n * ACTIVE_HEIGHT  # 10 units per key base
        max_height = ACTIVE_HEIGHT  # 10 units max
        
        # Get frequency weights
        freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        if scale_method == 'linear':
            weights = {k: v for k, v in freqs.items()}
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in freqs.items()}
        else:
            weights = {k: v for k, v in freqs.items()}
        
        total_weight = sum(weights.values())
        normalized = {k: w / total_weight for k, w in weights.items()}
        
        # === WIDTH CALCULATION (matching generate_keyboard_2d) ===
        widths = {k: int(round(normalized[k] * total_key_width)) for k in row_keys}
        
        # Enforce minimum width
        deficit = 0
        for letter in row_keys:
            if widths[letter] < min_w:
                deficit += min_w - widths[letter]
                widths[letter] = min_w
        
        if deficit > 0:
            large_keys = [k for k in row_keys if widths[k] > min_w]
            large_total = sum(widths[k] for k in large_keys)
            if large_total > 0:
                for letter in large_keys:
                    reduction = int(round(deficit * (widths[letter] / large_total)))
                    widths[letter] = max(min_w, widths[letter] - reduction)
        
        # === HEIGHT CALCULATION (matching generate_keyboard_2d) ===
        heights = {k: int(round(normalized[k] * total_height_budget)) for k in row_keys}
        
        # Cap at max height
        for letter in row_keys:
            heights[letter] = min(heights[letter], max_height)
        
        # Enforce minimum height
        for letter in row_keys:
            heights[letter] = max(heights[letter], min_h)
        
        # === ADD PADDING (2 units: 1 on each side) ===
        # The slot includes padding, so add 2 to width and height
        for letter in row_keys:
            widths[letter] += 2  # 1 unit padding each side
            heights[letter] += 2  # 1 unit padding top/bottom
        
        # === POSITION KEYS ===
        x = ROW_OFFSETS_UNITS[row_idx]
        row_top = row_idx * (SLOT_HEIGHT + ROW_GAP)
        
        for letter in row_keys:
            w = widths[letter]
            h = heights[letter]
            y = row_top  # Start at top of row slot
            
            key_bounds[letter] = (x, y, w, h)
            
            # Assign cells to this key
            for dy in range(h):
                for dx in range(w):
                    if 0 <= y + dy < grid_height and 0 <= x + dx < grid_width:
                        grid[y + dy][x + dx] = letter
            
            x += w  # Next key immediately after (spacing via padding)
    
    return UnitGridKeyboard(
        grid=grid,
        width=grid_width,
        height=grid_height,
        key_bounds=key_bounds
    )


def calculate_target_areas(scale_method: str = 'log') -> Dict[str, int]:
    """
    Calculate target area (in units) for each key based on frequency.
    
    Uses frequency-weighted redistribution within each row.
    Total area per row is conserved (row_keys × SLOT_WIDTH × SLOT_HEIGHT).
    
    Returns:
        Dict of letter -> target_units
    """
    target_areas = {}
    
    for row_idx, row_keys in enumerate(ROWS):
        n = len(row_keys)
        row_total_units = n * SLOT_WIDTH * SLOT_HEIGHT  # e.g., 10 * 108 = 1080 for row 0
        
        # Get frequency weights
        freqs = {letter: FREQUENCIES[letter] for letter in row_keys}
        
        if scale_method == 'linear':
            weights = freqs
        elif scale_method == 'sqrt':
            weights = {k: np.sqrt(v) for k, v in freqs.items()}
        elif scale_method == 'log':
            weights = {k: np.log1p(v) for k, v in freqs.items()}
        else:
            weights = freqs
        
        total_weight = sum(weights.values())
        
        # Distribute units proportionally
        for letter in row_keys:
            proportion = weights[letter] / total_weight
            target = int(round(proportion * row_total_units))
            # Ensure minimum viable size (at least 3×3 active = 5×5 total)
            target = max(target, 25)
            target_areas[letter] = target
    
    return target_areas


def align_keys_vertically(unit_kb: UnitGridKeyboard) -> UnitGridKeyboard:
    """
    Align keys vertically within their row slots.
    
    - Row 0 (top): Keys align to TOP of their row slot
    - Row 1 (middle): Keys align to CENTER of their row slot  
    - Row 2 (bottom): Keys align to BOTTOM of their row slot
    
    This should be applied after sizing but before fill_space expansion.
    
    Args:
        unit_kb: UnitGridKeyboard with sized keys
    
    Returns:
        New UnitGridKeyboard with vertically aligned keys
    """
    grid = [row[:] for row in unit_kb.grid]
    width, height = unit_kb.width, unit_kb.height
    
    # Row boundaries (including gaps between rows)
    row_stride = SLOT_HEIGHT + ROW_GAP  # Distance from one row start to next
    row_starts = [i * row_stride for i in range(3)]  # [0, 13, 26] with ROW_GAP=1
    row_ends = [start + SLOT_HEIGHT for start in row_starts]  # [12, 25, 38]
    
    for letter in FREQUENCIES:
        # Find all cells for this key
        cells = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == letter]
        if not cells:
            continue
        
        # Get current bounds
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        key_height = y_max - y_min + 1
        key_width = x_max - x_min + 1
        
        # Determine which row this key is in (based on original position)
        # Keys are assigned to rows based on which row slot they started in
        row_idx = y_min // row_stride
        row_idx = min(row_idx, 2)  # Clamp to valid row
        
        row_top = row_starts[row_idx]
        row_bottom = row_ends[row_idx]
        row_slot_height = row_bottom - row_top
        
        # Calculate new y position based on alignment
        if row_idx == 0:
            # Top row: align to top
            new_y_min = row_top
        elif row_idx == 1:
            # Middle row: align to center
            new_y_min = row_top + (row_slot_height - key_height) // 2
        else:
            # Bottom row: align to bottom
            new_y_min = row_bottom - key_height
        
        # Only move if position changes
        y_offset = new_y_min - y_min
        if y_offset == 0:
            continue
        
        # Clear old positions
        for x, y in cells:
            grid[y][x] = None
        
        # Set new positions
        for x, y in cells:
            new_y = y + y_offset
            if 0 <= new_y < height:
                grid[new_y][x] = letter
    
    # Rebuild key_bounds
    new_bounds = {}
    for letter in FREQUENCIES:
        cells = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == letter]
        if cells:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            new_bounds[letter] = (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        else:
            new_bounds[letter] = (0, 0, 0, 0)
    
    return UnitGridKeyboard(
        grid=grid,
        width=width,
        height=height,
        key_bounds=new_bounds
    )


def _get_neighbor_graph(unit_kb: UnitGridKeyboard) -> Dict[str, set]:
    """
    Build the neighbor graph: which keys share a border with which other keys.
    
    Checks all 8 directions (including diagonals) for adjacency.
    
    Returns:
        Dict of letter -> set of neighbor letters
    """
    grid = unit_kb.grid
    width, height = unit_kb.width, unit_kb.height
    
    neighbors = {letter: set() for letter in FREQUENCIES}
    
    for y in range(height):
        for x in range(width):
            letter = grid[y][x]
            if not letter:
                continue
            
            # Check all 8 adjacent cells (including diagonals)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = grid[ny][nx]
                        if neighbor and neighbor != letter:
                            neighbors[letter].add(neighbor)
    
    return neighbors


def fill_space(unit_kb: UnitGridKeyboard, 
               scale_method: str = 'log',
               max_iterations: int = 500) -> UnitGridKeyboard:
    """
    Fill empty space in a unit grid keyboard.
    
    RULES:
    1. Key = units + padding (all cells owned by key)
    2. Keys cannot touch (must maintain gap - no adjacent cells between different keys)
    3. Keys only grow (Log 2D sizes are minimums)
    4. Fixed boundaries (only claim cells within uniform grid territory)
    5. Frequency priority (high-freq keys claim first)
    6. Unclaimed = blank (cells that can't be claimed stay empty as padding)
    7. No direction preference
    8. Diagonal adjacency counts (8-way check for touching)
    
    Args:
        unit_kb: UnitGridKeyboard (typically from generate_unit_grid_log2d + align)
        scale_method: Not used (kept for API compatibility)
        max_iterations: Safety limit
    
    Returns:
        New UnitGridKeyboard with empty space filled (gaps remain as padding)
    """
    # Record original neighbor graph BEFORE any changes
    original_neighbors = _get_neighbor_graph(unit_kb)
    
    # Deep copy the grid
    grid = [row[:] for row in unit_kb.grid]
    width, height = unit_kb.width, unit_kb.height
    
    # RULE 4: Track which cells are within valid keyboard territory
    # (Based on UNIFORM grid - the maximum extent of each row)
    # Row 0: x = 0 to 89 (10 keys * 9 units)
    # Row 1: x = 4 to 84 (offset 4, 9 keys * 9 units)  
    # Row 2: x = 8 to 70 (offset 8, 7 keys * 9 units)
    valid_territory = [[False for _ in range(width)] for _ in range(height)]
    for row_idx in range(len(ROWS)):
        row_start_x = ROW_OFFSETS_UNITS[row_idx]
        row_end_x = row_start_x + len(ROWS[row_idx]) * SLOT_WIDTH
        row_start_y = row_idx * (SLOT_HEIGHT + ROW_GAP)
        row_end_y = row_start_y + SLOT_HEIGHT
        
        for y in range(row_start_y, min(row_end_y, height)):
            for x in range(row_start_x, min(row_end_x, width)):
                valid_territory[y][x] = True
    
    def get_cells(letter):
        return [(x, y) for y in range(height) for x in range(width) if grid[y][x] == letter]
    
    def is_adjacent_to_key(x, y, letter):
        """Check if cell (x,y) is adjacent to any cell owned by letter."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny][nx] == letter:
                    return True
        return False
    
    def would_touch_another_key(letter, x, y):
        """
        Check if claiming cell (x,y) would make this key directly touch
        another key (no gap between them).
        
        To maintain visual padding, we don't allow claiming cells that are
        adjacent to any other key's cells - even original neighbors.
        
        Checks all 8 directions (including diagonals).
        """
        # Check all 8 directions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    adj_letter = grid[ny][nx]
                    if adj_letter and adj_letter != letter:
                        return True  # Would touch another key!
        return False
    
    
    def can_claim_cell(letter, x, y):
        """Check if 'letter' can validly claim cell (x,y)."""
        # Cell must be empty
        if grid[y][x] is not None:
            return False
        
        # RULE 4: Cell must be within original keyboard territory
        if not valid_territory[y][x]:
            return False
        
        # Must be adjacent to existing cells of this key
        if not is_adjacent_to_key(x, y, letter):
            return False
        
        # Must not touch another key (maintain padding gap)
        if would_touch_another_key(letter, x, y):
            return False
        
        return True
    
    def get_claimable_cells(letter):
        """Get all empty cells adjacent to this key that it can validly claim."""
        cells = get_cells(letter)
        claimable = set()
        
        for cx, cy in cells:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if can_claim_cell(letter, nx, ny):
                        claimable.add((nx, ny))
        
        return list(claimable)
    
    # Sort keys by frequency (highest first for claiming priority)
    sorted_by_freq = sorted(FREQUENCIES.keys(), key=lambda k: FREQUENCIES[k], reverse=True)
    
    # Iteratively fill empty cells
    for iteration in range(max_iterations):
        changes_made = False
        
        # Each key tries to claim adjacent empty cells
        for letter in sorted_by_freq:
            claimable = get_claimable_cells(letter)
            
            for x, y in claimable:
                # Double-check cell is still empty (might have been claimed)
                if grid[y][x] is None and can_claim_cell(letter, x, y):
                    grid[y][x] = letter
                    changes_made = True
        
        if not changes_made:
            break
    
    # Rule 6: Leave unclaimed cells as None (blank)
    # (No Phase 3 cleanup - if no one can claim it, it stays empty)
    
    # Rebuild key_bounds
    new_bounds = {}
    for letter in FREQUENCIES:
        cells = get_cells(letter)
        if cells:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            new_bounds[letter] = (min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        else:
            new_bounds[letter] = (0, 0, 0, 0)
    
    return UnitGridKeyboard(
        grid=grid,
        width=width,
        height=height,
        key_bounds=new_bounds
    )


def visualize_unit_grid(unit_kb: UnitGridKeyboard, title: str = "Unit Grid Keyboard",
                        show_padding: bool = True):
    """
    Visualize a unit grid keyboard.
    
    Args:
        unit_kb: UnitGridKeyboard to visualize
        title: Plot title
        show_padding: If True, show padding cells in lighter color
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Color map for keys
    colors = {}
    for letter in FREQUENCIES:
        # Color based on frequency (high = warm, low = cool)
        freq = FREQUENCIES[letter]
        max_freq = max(FREQUENCIES.values())
        norm_freq = freq / max_freq
        
        # Blue (low freq) to Red (high freq)
        colors[letter] = (0.2 + 0.6 * norm_freq, 0.3, 0.8 - 0.6 * norm_freq, 0.9)
    
    # Draw each cell
    for y in range(unit_kb.height):
        for x in range(unit_kb.width):
            letter = unit_kb.grid[y][x]
            if letter is None:
                continue
            
            color = colors[letter]
            
            # Check if this is a padding cell (on the edge of the key's bounds)
            if show_padding:
                cells = unit_kb.get_key_cells(letter)
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                is_padding = (x == x_min or x == x_max or y == y_min or y == y_max)
                if is_padding:
                    # Lighter color for padding
                    color = (color[0] * 0.5 + 0.5, color[1] * 0.5 + 0.5, color[2] * 0.5 + 0.5, 0.5)
            
            rect = patches.Rectangle(
                (x, y), 1, 1,
                linewidth=0.2,
                edgecolor='gray',
                facecolor=color
            )
            ax.add_patch(rect)
    
    # Add key labels at centroids
    for letter in FREQUENCIES:
        cells = unit_kb.get_key_cells(letter)
        if cells:
            cx = sum(c[0] for c in cells) / len(cells) + 0.5
            cy = sum(c[1] for c in cells) / len(cells) + 0.5
            ax.text(cx, cy, letter, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    ax.set_xlim(0, unit_kb.width)
    ax.set_ylim(unit_kb.height, 0)  # Invert Y
    ax.set_aspect('equal')
    ax.set_xlabel('units (mm)')
    ax.set_ylabel('units (mm)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def compare_unit_grids(keyboards: List[Tuple[str, UnitGridKeyboard]]):
    """Compare multiple unit grid keyboards side by side."""
    n = len(keyboards)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
    
    if n == 1:
        axes = [axes]
    
    colors = {}
    for letter in FREQUENCIES:
        freq = FREQUENCIES[letter]
        max_freq = max(FREQUENCIES.values())
        norm_freq = freq / max_freq
        colors[letter] = (0.2 + 0.6 * norm_freq, 0.3, 0.8 - 0.6 * norm_freq, 0.9)
    
    for ax, (title, unit_kb) in zip(axes, keyboards):
        for y in range(unit_kb.height):
            for x in range(unit_kb.width):
                letter = unit_kb.grid[y][x]
                if letter is None:
                    continue
                
                color = colors[letter]
                
                # Padding detection
                cells = unit_kb.get_key_cells(letter)
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                is_padding = (x == x_min or x == x_max or y == y_min or y == y_max)
                if is_padding:
                    color = (color[0] * 0.5 + 0.5, color[1] * 0.5 + 0.5, color[2] * 0.5 + 0.5, 0.5)
                
                rect = patches.Rectangle(
                    (x, y), 1, 1,
                    linewidth=0.1,
                    edgecolor='gray',
                    facecolor=color
                )
                ax.add_patch(rect)
        
        # Labels
        for letter in FREQUENCIES:
            cells = unit_kb.get_key_cells(letter)
            if cells:
                cx = sum(c[0] for c in cells) / len(cells) + 0.5
                cy = sum(c[1] for c in cells) / len(cells) + 0.5
                ax.text(cx, cy, letter, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white')
        
        ax.set_xlim(0, unit_kb.width)
        ax.set_ylim(unit_kb.height, 0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def print_unit_grid_stats(unit_kb: UnitGridKeyboard, name: str):
    """Print statistics for a unit grid keyboard."""
    print(f"\n{name}:")
    print("-" * 50)
    
    areas = [(letter, unit_kb.get_key_area(letter)) for letter in FREQUENCIES]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    total_area = sum(a for _, a in areas)
    uniform_area = len(FREQUENCIES) * SLOT_WIDTH * SLOT_HEIGHT
    
    print(f"  Grid size: {unit_kb.width} × {unit_kb.height} = {unit_kb.width * unit_kb.height} units")
    print(f"  Total assigned: {total_area} units (uniform: {uniform_area})")
    print()
    print("  Top 5 largest keys:")
    for letter, area in areas[:5]:
        active = unit_kb.get_active_bounds(letter)
        print(f"    {letter}: {area} units ({area/108:.2f}x uniform), active: {active[2]}×{active[3]}")
    print()
    print("  Top 5 smallest keys:")
    for letter, area in areas[-5:]:
        active = unit_kb.get_active_bounds(letter)
        print(f"    {letter}: {area} units ({area/108:.2f}x uniform), active: {active[2]}×{active[3]}")


def generate_uniform_grid_keyboard() -> List[GridKey]:
    """Generate a uniform grid keyboard (all keys are rectangles with padding)."""
    key_info = {}
    
    for row_idx, row_keys in enumerate(ROWS):
        x = GRID_ROW_OFFSETS[row_idx]
        for letter in row_keys:
            key_info[letter] = {
                'row': row_idx,
                'cols': list(range(x, x + GRID_KEY_WIDTH))
            }
            x += GRID_KEY_WIDTH
    
    # Row pitch includes padding between rows
    row_pitch = GRID_ROW_HEIGHT + PADDING
    
    grid_keys = []
    for letter, info in key_info.items():
        row_idx = info['row']
        # Top of row with padding accounted for
        row_top = row_idx * row_pitch
        # Leave padding at bottom (half padding top, half bottom = full padding between rows)
        row_bottom = row_top + GRID_ROW_HEIGHT
        
        column_bounds = {col: (row_top, row_bottom) for col in info['cols']}
        
        grid_keys.append(GridKey(
            letter=letter,
            frequency=FREQUENCIES[letter],
            row=row_idx,
            column_bounds=column_bounds
        ))
    
    return grid_keys


# =============================================================================
# Visualization
# =============================================================================

def get_color_for_scale(scale: float) -> Tuple[float, float, float, float]:
    """
    Generate a color based on scale relative to uniform.
    - scale < 1: blue (shrunk)
    - scale = 1: neutral
    - scale > 1: red/orange (grown)
    """
    if scale < 1:
        intensity = scale
        return (0.2, 0.3, 0.5 + 0.5 * intensity, 0.8)
    else:
        intensity = min((scale - 1) / 1.5, 1)
        return (0.8 + 0.2 * intensity, 0.4 - 0.2 * intensity, 0.2, 0.8)


def visualize_keyboard(keys: List[Key], title: str = "Keyboard Layout", 
                       show_scale: bool = True, color_by: str = 'scale'):
    """Render a single keyboard layout."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    max_freq = max(k.frequency for k in keys)
    
    for key in keys:
        if color_by == 'scale':
            color = get_color_for_scale(key.scale)
        else:
            normalized = key.frequency / max_freq
            color = (normalized, 0.3, 1 - normalized, 0.7)
        
        rect = patches.FancyBboxPatch(
            (key.left, key.bottom),
            key.width,
            key.height,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            linewidth=2,
            edgecolor='black',
            facecolor=color
        )
        ax.add_patch(rect)
        
        ax.text(key.center_x, key.center_y + 1, key.letter,
                ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        if show_scale:
            ax.text(key.center_x, key.center_y - 2, f"{key.scale:.2f}x",
                    ha='center', va='center',
                    fontsize=7, color='white', alpha=0.9)
    
    ax.text(0.02, 0.98, "🔵 Shrunk (<1x)  🔴 Grown (>1x)",
           transform=ax.transAxes, fontsize=9, color='gray',
           verticalalignment='top')
    
    ax.set_xlim(-5, 85)
    ax.set_ylim(-30, 15)
    ax.set_aspect('equal')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def compare_keyboards(keyboards: List[Tuple[str, List[Key]]]):
    """Show multiple keyboard layouts side by side."""
    n = len(keyboards)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    
    if n == 1:
        axes = [axes]
    
    # Find global bounds across all keyboards for consistent scaling
    all_keys = [k for _, keys in keyboards for k in keys]
    min_y = min(k.bottom for k in all_keys) - 5
    max_y = max(k.top for k in all_keys) + 5
    
    for ax, (title, keys) in zip(axes, keyboards):
        for key in keys:
            color = get_color_for_scale(key.scale)
            rect = patches.FancyBboxPatch(
                (key.left, key.bottom),
                key.width,
                key.height,
                boxstyle="round,pad=0.02,rounding_size=0.5",
                linewidth=1.5,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(rect)
            ax.text(key.center_x, key.center_y, key.letter,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        
        ax.set_xlim(-5, 85)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    axes[0].text(0.02, 0.98, "🔵 Shrunk  🔴 Grown",
                transform=axes[0].transAxes, fontsize=8, color='gray',
                verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def verify_row_widths(keys: List[Key], name: str):
    """Verify that total key width per row stays constant."""
    print(f"\n{name}:")
    print("-" * 40)
    
    for row_idx, row_letters in enumerate(ROWS):
        row_keys = [k for k in keys if k.letter in row_letters]
        total_width = sum(k.width for k in row_keys)
        expected = len(row_letters) * BASE_WIDTH
        status = '✓' if abs(total_width - expected) < 0.01 else '✗'
        print(f"  Row {row_idx}: {total_width:.2f}mm (expected: {expected:.1f}mm) {status}")


# =============================================================================
# Grid Keyboard Visualization
# =============================================================================

def get_color_for_grid_scale(scale: float) -> Tuple[float, float, float, float]:
    """
    Generate a color based on scale relative to uniform.
    Similar to get_color_for_scale but tuned for grid visualization.
    """
    if scale < 1:
        # Blue for shrunk keys
        intensity = max(0.2, scale)
        return (0.15, 0.25, 0.4 + 0.5 * intensity, 0.85)
    else:
        # Orange/red for grown keys
        intensity = min((scale - 1) / 1.0, 1)
        return (0.85 + 0.15 * intensity, 0.45 - 0.3 * intensity, 0.15, 0.85)


def visualize_grid_keyboard(grid_keys: List[GridKey], title: str = "Grid Keyboard"):
    """
    Render a grid-based keyboard with polygon-shaped keys.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for key in grid_keys:
        if not key.column_bounds:
            continue
        
        # Get polygon points
        points = key.get_polygon_points()
        if len(points) < 3:
            continue
        
        # Color based on scale
        color = get_color_for_grid_scale(key.scale)
        
        # Draw polygon
        polygon = plt.Polygon(
            points,
            closed=True,
            facecolor=color,
            edgecolor='black',
            linewidth=0.8
        )
        ax.add_patch(polygon)
        
        # Label at centroid
        cx, cy = key.centroid
        ax.text(cx, cy, key.letter,
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Set axis limits based on actual key bounds
    row_pitch = GRID_ROW_HEIGHT + PADDING
    total_height = len(ROWS) * row_pitch
    max_col = max(max(key.column_bounds.keys()) for key in grid_keys if key.column_bounds)
    
    ax.set_xlim(-2, max_col + 5)
    ax.set_ylim(total_height + 2, -2)  # Inverted Y (0 at top)
    ax.set_aspect('equal')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    ax.text(0.02, 0.98, "🔵 Shrunk (<1x)  🔴 Grown (>1x)",
            transform=ax.transAxes, fontsize=9, color='gray',
            verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def compare_grid_keyboards(keyboards: List[Tuple[str, List[GridKey]]]):
    """
    Show multiple grid keyboard layouts side by side.
    """
    n = len(keyboards)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    
    if n == 1:
        axes = [axes]
    
    # Find global bounds
    row_pitch = GRID_ROW_HEIGHT + PADDING
    total_height = len(ROWS) * row_pitch
    max_col = 0
    for _, keys in keyboards:
        for key in keys:
            if key.column_bounds:
                max_col = max(max_col, max(key.column_bounds.keys()))
    
    for ax, (title, grid_keys) in zip(axes, keyboards):
        for key in grid_keys:
            if not key.column_bounds:
                continue
            
            points = key.get_polygon_points()
            if len(points) < 3:
                continue
            
            color = get_color_for_grid_scale(key.scale)
            
            polygon = plt.Polygon(
                points,
                closed=True,
                facecolor=color,
                edgecolor='black',
                linewidth=0.6
            )
            ax.add_patch(polygon)
            
            cx, cy = key.centroid
            ax.text(cx, cy, key.letter,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        
        ax.set_xlim(-2, max_col + 5)
        ax.set_ylim(total_height + 2, -2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)
    
    axes[0].text(0.02, 0.98, "🔵 Shrunk  🔴 Grown",
                 transform=axes[0].transAxes, fontsize=8, color='gray',
                 verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def print_grid_stats(grid_keys: List[GridKey], name: str):
    """Print statistics for a grid keyboard."""
    print(f"\n{name}:")
    print("-" * 50)
    
    # Sort by area
    sorted_keys = sorted(grid_keys, key=lambda k: k.area, reverse=True)
    
    total_area = sum(k.area for k in sorted_keys)
    uniform_total = len(sorted_keys) * BASE_WIDTH * BASE_HEIGHT
    
    print(f"  Total area: {total_area:.1f} mm² (uniform: {uniform_total:.1f} mm²)")
    print(f"  Area ratio: {total_area/uniform_total:.2%}")
    print()
    print("  Top 5 largest keys:")
    for k in sorted_keys[:5]:
        print(f"    {k.letter}: {k.area:.1f} mm² ({k.scale:.2f}x)")
    print()
    print("  Top 5 smallest keys:")
    for k in sorted_keys[-5:]:
        print(f"    {k.letter}: {k.area:.1f} mm² ({k.scale:.2f}x)")

