import numpy as np
import cv2
import random
from PIL import Image, ImageDraw, ImageFilter


class HistologyArtifactGenerator:
    def __init__(self, size=512):
        self.size = size

        # Color palettes (H&E typical colors)
        # Deep Hematoxylin Purple / Blue
        # self.colors_hema = [
        #    (70, 20, 70),   # Deep Purple
        #    (40, 10, 60),   # Very Dark Purple
        #    (60, 60, 100),  # Bluish
        # ]
        self.colors_hema = [
            (70, 20, 70),  # Deep Purple
            (30, 0, 60),  # <--- NEW: Deep Indigo (Real Crystal Color)
            (20, 0, 40),
        ]
        # Eosin Pink / Red
        self.colors_eosin = [
            (180, 50, 80),  # Deep Red
            (200, 100, 120),  # Pinkish
            (150, 20, 40),  # Blood red
        ]

    # ==========================================
    # 1. SHAPE GENERATION
    # ==========================================
    def generate_polygon_mask(self):
        """Generates crystalline/geometric shapes (Voronoi-like)"""
        mask = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(mask)

        # Random center
        cx, cy = (
            random.randint(100, self.size - 100),
            random.randint(100, self.size - 100),
        )
        radius = random.randint(40, 120)

        # Generate random points in a circle
        points = []
        num_points = random.randint(5, 12)
        for _ in range(num_points):
            angle = random.uniform(0, 2 * np.pi)
            r = radius * random.uniform(0.5, 1.0)
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            points.append((px, py))

        # Draw Convex Hull to simulate crystal
        # Use OpenCV for convex hull calculation
        pts_np = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts_np)

        # Convert back to list of tuples for PIL
        hull_points = [tuple(pt[0]) for pt in hull]

        draw.polygon(hull_points, fill=255)
        return mask

    def generate_blob_mask(self):
        """Generates fluid-like blobs (metabolites/dye pools)"""
        mask = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(mask)

        cx, cy = (
            random.randint(100, self.size - 100),
            random.randint(100, self.size - 100),
        )

        # Draw multiple overlapping circles to create an amorphous blob
        num_circles = random.randint(5, 15)
        for _ in range(num_circles):
            dx = random.randint(-40, 40)
            dy = random.randint(-40, 40)
            r = random.randint(20, 60)
            draw.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r], fill=255)

        # Blur heavily to merge them into a fluid shape, then threshold back
        mask = mask.filter(ImageFilter.GaussianBlur(10))

        # Thresholding to sharpen edges slightly but keep them organic
        mask_arr = np.array(mask)
        mask_arr = np.where(mask_arr > 100, 255, 0).astype(np.uint8)

        return Image.fromarray(mask_arr)

    def generate_fiber_mask(self):
        """Generates thick, striated fiber artifacts (threads/folds)"""
        mask = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(mask)

        # Random curve points (Bezier-like spine)
        # We start well outside the image to ensure it crosses fully
        if random.random() < 0.5:  # Vertical-ish
            x_start = random.randint(0, self.size)
            y_start = -50
            x_end = x_start + random.randint(-100, 100)
            y_end = self.size + 50
        else:  # Horizontal-ish
            x_start = -50
            y_start = random.randint(0, self.size)
            x_end = self.size + 50
            y_end = y_start + random.randint(-100, 100)

        # Create a simple 2-point line for the main spine (or add midpoints for curve)
        # For simple thick fibers, a straight-ish line is often enough
        points = [(x_start, y_start), (x_end, y_end)]

        # 1. Draw the Main Body (Thick)
        # Real fibers are HUGE (40-100px wide)
        main_width = random.randint(40, 100)

        # Draw the base opacity (not full 255, to allow striations to pop)
        draw.line(points, fill=180, width=main_width)

        # 2. Add Striations (Internal texture)
        # Draw thinner, brighter lines running parallel to the main line
        num_strands = random.randint(5, 12)
        for _ in range(num_strands):
            # Offset the start/end perpendicular to the line?
            # Easier way: Just jitter the coordinates slightly
            offset = random.randint(-main_width // 2, main_width // 2)

            # Simple offset logic (works well for vertical/horizontal)
            if abs(x_start - x_end) < abs(y_start - y_end):  # Vertical
                s_p = (x_start + offset, y_start)
                e_p = (x_end + offset, y_end)
            else:  # Horizontal
                s_p = (x_start, y_start + offset)
                e_p = (x_end, y_end + offset)

            strand_width = random.randint(2, 5)
            # These strands are fully opaque (255) creating "ribs" inside the fiber
            draw.line([s_p, e_p], fill=255, width=strand_width)

        # 3. Blur slightly to merge the strands into a cohesive object
        mask = mask.filter(ImageFilter.GaussianBlur(2))

        return mask

    def generate_granular_mask(self):
        """
        Generates:
        1. Texture Mask: Individual scattered dots (Physical dirt).
        2. Annotation Mask: A Convex Hull Polygon wrapping the dots (Human Label).
        """
        # 1. Canvas for the Texture (The actual pixels to generate)
        texture_mask = Image.new("L", (self.size, self.size), 0)
        draw_tex = ImageDraw.Draw(texture_mask)

        # 2. Canvas for the Annotation (The label for the model)
        annotation_mask = Image.new("L", (self.size, self.size), 0)
        draw_ann = ImageDraw.Draw(annotation_mask)

        # Center of the clump
        cx, cy = (
            random.randint(100, self.size - 100),
            random.randint(100, self.size - 100),
        )

        # Sprawling cluster parameters
        num_particles = random.randint(50, 150)
        cluster_radius = random.randint(40, 100)

        all_points = []

        for _ in range(num_particles):
            # Biased distribution towards center, but spread out
            angle = random.uniform(0, 2 * np.pi)
            dist = random.triangular(0, cluster_radius, 0)

            px = int(cx + dist * np.cos(angle))
            py = int(cy + dist * np.sin(angle))

            # Keep within bounds
            px = np.clip(px, 0, self.size - 1)
            py = np.clip(py, 0, self.size - 1)

            # Particle size
            r = random.randint(2, 6)

            # Draw the physical dirt speck on the texture mask
            draw_tex.ellipse([px - r, py - r, px + r, py + r], fill=255)

            # Collect points for the hull calculation
            # We add points for the edges of the particle to ensure the hull covers it fully
            all_points.append([px - r, py - r])
            all_points.append([px + r, py + r])
            all_points.append([px - r, py + r])
            all_points.append([px + r, py - r])

        # --- APPLY CONVEX HULL FOR POLYGON ANNOTATION ---
        if len(all_points) > 0:
            pts_np = np.array(all_points, dtype=np.int32)

            # Compute the convex hull (the "rubber band" around the points)
            hull = cv2.convexHull(pts_np)

            # Convert to list of tuples for PIL
            hull_points = [tuple(pt[0]) for pt in hull]

            # Draw the polygon
            draw_ann.polygon(hull_points, fill=255)

        return annotation_mask, texture_mask

    # ==========================================
    # 2. TEXTURE GENERATION
    # ==========================================
    def create_artifact_texture(self, shape_mask, color_base):
        """
        Creates the gritty, noisy texture inside the artifact.
        """
        # 1. Create base noise layer
        # Standard Gaussian noise
        noise_arr = np.random.normal(0, 30, (self.size, self.size, 3)).astype(np.uint8)
        noise_img = Image.fromarray(noise_arr, mode="RGB")

        # 2. Create solid color layer
        color_layer = Image.new("RGB", (self.size, self.size), color_base)

        # 3. Blend solid color with noise (Overlay mode equivalent)
        # This gives the color some "grit"
        texture = Image.blend(color_layer, noise_img, alpha=0.1)

        # 4. Apply Mask
        # We start with a white canvas (transparent in Multiply mode)
        final_artifact = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        final_artifact.paste(texture, (0, 0), shape_mask)

        return final_artifact

    # ==========================================
    # 3. PHYSICS-BASED BLENDING
    # ==========================================
    def blend_multiply(self, tissue, artifact, mask, opacity=0.85):
        """
        Simulates light passing through two layers (Tissue + Stain).
        Math: Result = (Tissue * Artifact) / 255
        """
        tissue_arr = np.array(tissue).astype(float)
        artifact_arr = np.array(artifact).astype(float)
        mask_arr = np.array(mask).astype(float) / 255.0

        # 1. Normalize Artifact:
        # 255 (White) -> 1.0 (No change)
        # 0 (Black)   -> 0.0 (Full darkness)
        artifact_norm = artifact_arr / 255.0

        # 2. Apply Opacity to the Artifact itself before blending
        # If opacity is 0.5, the artifact moves halfway towards 1.0 (White/Transparent)
        # Logic: FinalArtifact = Artifact * Opacity + 1.0 * (1-Opacity)
        # Actually easier: Just blend the mask intensity.

        # 3. The Multiply Blend
        # We want: Where mask is 0, Multiplier is 1.0. Where mask is 1, Multiplier is Artifact.
        # Multiplier = 1.0 - (mask * (1.0 - artifact_norm) * opacity)

        # Calculate the "Darkening Factor"
        # If artifact is purple (0.5, 0.0, 0.5), it darkens green channel heavily.
        inverse_artifact = 1.0 - artifact_norm
        weighted_effect = inverse_artifact * mask_arr[:, :, None] * opacity
        multiplier = 1.0 - weighted_effect

        # Apply
        blended = tissue_arr * multiplier

        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def apply_refraction(self, tissue, mask, strength=5.0):
        """
        Simulates light bending through a crystal by displacing pixels.
        """
        img_arr = np.array(tissue)
        rows, cols = img_arr.shape[:2]

        # FIX: Use meshgrid to correctly create coordinate maps of shape (rows, cols)
        map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Find indices where the mask is active
        # Ensure mask is a numpy array (single channel)
        mask_arr = np.array(mask)
        mask_indices = np.where(mask_arr > 0)

        # Apply displacement only if there is a mask
        if len(mask_indices[0]) > 0:
            # Shift X and Y coordinates inside the mask
            shift_x = np.random.uniform(-strength, strength)
            shift_y = np.random.uniform(-strength, strength)

            # Now we can safely index because maps match image dimensions
            map_x[mask_indices] += shift_x
            map_y[mask_indices] += shift_y

            # Remap image (Distort the tissue)
            distorted = cv2.remap(img_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            return Image.fromarray(distorted)

        return tissue

    # ==========================================
    # 4. MAIN PIPELINE
    # ==========================================
    def apply_random_artifact(self, tissue_image):
        """
        Takes a PIL image (tissue), adds a random artifact.
        Returns: (Synthesized Image, Binary Mask)
        """
        tissue_image = tissue_image.resize((self.size, self.size))

        # 1. Pick a Type
        artifact_type = random.choice(["crystal", "blob", "fiber", "granular"])

        if artifact_type == "crystal":
            mask = self.generate_polygon_mask()
            color = random.choice(self.colors_hema)  # Crystals often Hematoxylin
            opacity = random.uniform(0.6, 0.9)  # Crystals are darker/denser
            tissue_image = self.apply_refraction(tissue_image, mask, strength=4.0)
        elif artifact_type == "blob":
            mask = self.generate_blob_mask()
            color = random.choice(self.colors_eosin)  # Blobs often Eosin
            opacity = random.uniform(0.4, 0.7)  # Blobs are often semi-transparent fluid
        elif artifact_type == "fiber":
            mask = self.generate_fiber_mask()
            color = random.choice([(50, 50, 50), (100, 80, 80), (80, 20, 60)])
            opacity = random.uniform(0.6, 0.85)
        else:
            mask, texture_pattern = self.generate_granular_mask()

            color = random.choice([(40, 30, 30), (80, 20, 20), (20, 20, 20)])
            opacity = random.uniform(0.85, 0.98)  # Dirt is dense

            # IMPORTANT: We use the *texture_pattern* to generate the pixels...
            # This creates the "dots"
            mask_blurred_for_texture = texture_pattern.filter(
                ImageFilter.GaussianBlur(radius=1)
            )
            artifact_texture = self.create_artifact_texture(
                mask_blurred_for_texture, color
            )

            # ...but we use the rough *mask* for the blend opacity and output
            # This ensures the model learns to predict the "blob", not the dots.

            # However, for the *visual* generation of the image, we only want dirt
            # where the dots are.
            result_img = self.blend_multiply(
                tissue_image, artifact_texture, mask_blurred_for_texture, opacity
            )

            # Return the ROUGH mask for the training label
            return result_img, mask

        # 2. Feather the mask edges (Critical for avoiding "Red Edge")
        # Soft edges allow the multiply blend to fade out naturally
        mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=2))

        # 3. Generate Texture
        artifact_texture = self.create_artifact_texture(mask_blurred, color)

        # 4. Blend
        result_img = self.blend_multiply(
            tissue_image, artifact_texture, mask_blurred, opacity
        )

        return result_img, mask


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    import sys

    fake_tissue = Image.open(sys.argv[1]).convert("RGB")

    # Load your clean tissue image
    # For testing, we can create a fake pink tissue background
    # fake_tissue = Image.new('RGB', (512, 512), (230, 200, 220))
    # Add some noise to fake tissue so we can see the transparency effect
    # noise = np.random.normal(0, 10, (512, 512, 3)).astype(np.uint8)
    # fake_tissue = Image.fromarray(np.clip(np.array(fake_tissue) + noise, 0, 255).astype(np.uint8))

    generator = HistologyArtifactGenerator(size=756)

    # Generate 5 examples
    for i in range(5):
        syn_image, syn_mask = generator.apply_random_artifact(fake_tissue)

        syn_image.save(f"synthetic_artifact_{i}.jpg")
        syn_mask.save(f"synthetic_mask_{i}.png")
        print(f"Saved synthetic_artifact_{i}.jpg")

    print("Done! Check the files.")
