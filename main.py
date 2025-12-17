import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_images(path_name):
    """Loads images from the specified path and returns them as a numpy array.

    Args:
        path_name (str): Path to the image file.
    Returns:
        numpy.ndarray: 2D array of shape (height, width) containing the images.
    """
    image = Image.open(path_name)
    image = np.asarray(image)
    if image.ndim == 3:
        # Convert to grayscale if RGB
        image = image[:, :, 0]
    return image

def chunk_image(image, chunk_size):
    """Chunks the input image into smaller patches of size chunk_size x chunk_size.

    Args:
        image (numpy.ndarray): 2D array representing the image to be chunked.
        chunk_size (int): Size of each chunk (both width and height).
    Returns:
        numpy.ndarray: 4D array of shape (num_chunks_y, num_chunks_x, chunk_size, chunk_size)
                       containing the image chunks.
    """
    img_height, img_width = image.shape
    chunks_y = img_height // chunk_size
    chunks_x = img_width // chunk_size

    if chunks_y == 0 or chunks_x == 0:
        raise ValueError("Chunk size is larger than the image dimensions.")

    if img_height % chunk_size != 0 or img_width % chunk_size != 0:
        print("Warning: Image dimensions are not perfectly divisible by chunk size. "
              "Trimming the image to fit.")
        # Trim the image to make it divisible by chunk_size
        image = image[:chunks_y * chunk_size, :chunks_x * chunk_size]

    # Reshape and transpose to get the chunks
    chunk_array = image.reshape(chunks_y, chunk_size, chunks_x, chunk_size)
    chunk_array = chunk_array.transpose(0, 2, 1, 3)

    return chunk_array

def _find_random_defect_boxes(image, num_defects=None):
    """Generates random bounding boxes to simulate defect detection.

    Args:
        image (numpy.ndarray): 2D array representing the image chunk.
        num_defects (int): Number of random defects to generate.
    Returns:
        list: List of bounding boxes in the format (x_min, y_min, width, height, [rotation]).
    """
    if num_defects is None:
        num_defects = np.random.randint(0, 5)  # Randomly choose 1 to 3 defects
    height, width = image.shape
    boxes = []
    for _ in range(num_defects):
        box_width = np.random.randint(5, min(20, width // 2))
        box_height = np.random.randint(5, min(20, height // 2))
        x_min = np.random.randint(0, width - box_width)
        y_min = np.random.randint(0, height - box_height)
        rotation = np.random.uniform(0, 360)  # Random rotation angle
        boxes.append((x_min, y_min, box_width, box_height, rotation))
    return boxes

def find_defects(image):
    """Located the defects in the image chunk.
    Args:
        image (numpy.ndarray): 2D array of shape (height, width) representing the image chunk.
    Returns:
        list of bounding boxes for the detected defects in the format (x_min, y_min, width, height, rotation).
    """

    #### Placeholder defect detection logic that defines random boxes ###
    defects = _find_random_defect_boxes(image)
    return defects

def find_defects_array(image_array):
    """Locate defects in each image chunk of the input array.

    Args:
        image_array (numpy.ndarray): 4D array of shape (num_chunks_y, num_chunks_x, chunk_size, chunk_size)
                                     containing the image chunks.
    Returns:
        defects: An array of lists containing bounding boxes for detected defects in each chunk.
    """
    array_shape = image_array.shape[:2]
    defects = np.empty(array_shape, dtype=object)

    for i in np.ndindex(array_shape):
        chunk = image_array[i]
        defects[i] = find_defects(chunk)

    return defects

def transform_bounding_boxes(defects, chunk_size):
    """Transforms bounding boxes from chunk coordinates to original image coordinates.

    Args:
        defects (numpy.ndarray): 2D array of shape (num_chunks_y, num_chunks_x) containing lists of bounding boxes.
        chunk_size (int): Size of each chunk (both width and height).
    Returns:
        list: List of transformed bounding boxes in the format (x_min, y_min, width, height, [rotation]).
    """
    transformed_boxes = []
    num_chunks_y, num_chunks_x = defects.shape

    for i in range(num_chunks_y):
        for j in range(num_chunks_x):
            chunk_boxes = defects[i, j]
            for box in chunk_boxes:
                x_min, y_min, width, height, rotation = box
                # Transform coordinates to original image
                x_min_global = x_min + j * chunk_size
                y_min_global = y_min + i * chunk_size
                transformed_boxes.append((x_min_global, y_min_global, width, height, rotation))

    return transformed_boxes

def display_bounding_boxes(image, bounding_boxes):
    """Displays the image with bounding boxes overlaid.

    Args:
        image (numpy.ndarray): 2D array representing the original image.
        bounding_boxes (list): List of bounding boxes in the format (x_min, y_min, width, height, [rotation]).
    """

    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')

    for box in bounding_boxes:
        x_min, y_min, width, height, rotation = box
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 angle=rotation, linewidth=1,
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def main(path_name, chunk_size):
    # Load image
    image = load_images(path_name)

    # # Chunk image
    # image_chunks = chunk_image(image, chunk_size)

    # # Find defects in chunks
    # defects_in_chunks = find_defects_array(image_chunks)

    # # Transform bounding boxes to original image coordinates
    # bounding_boxes = transform_bounding_boxes(defects_in_chunks, chunk_size)

    bounding_boxes = find_defects(image)

    # Display bounding boxes on original image
    display_bounding_boxes(image, bounding_boxes)
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_image> <chunk_size>")
        sys.exit(1)

    path_name = sys.argv[1]
    chunk_size = int(sys.argv[2])

    main(path_name, chunk_size)