Bias in Face and Emotion Recognition: Methodological Considerations for Multicultural and Multimodal Research
---------------------------------------------------------

This Python script facilitates the analysis of skin tone in images for research on potential biases in face and emotion recognition systems. It leverages the von Luschan scale, a historical skin color classification system, to explore its limitations in the context of modern multicultural and multimodal research.


**Dependencies:**

-   Python 3 (tested with specific version, if applicable)
-   Pillow (PIL Fork) for image processing (`pip install Pillow`)
-   NumPy (`pip install numpy`)
-   SciPy (`pip install scipy`)
-   OpenCV (`pip install opencv-python`)
-   Pandas (`pip install pandas`)
-   tqdm (optional, for progress bar) (`pip install tqdm`)
-   matplotlib (optional, for visualization) (`pip install matplotlib`)
-   seaborn (optional, for visualization) (`pip install seaborn`)
-   scikit-image (optional) (`pip install scikit-image`)
-   Additional libraries might be required depending on specific functions used in the script (check the script for imports)

**Instructions:**

1.  **Clone or Download the Repository:** Obtain the script files for this project.

2.  **Prepare Data:**

    -   Create a directory named `img_directory` to store the images you want to analyze.
    -   Ensure all image files in this directory are in PNG format.
3.  **Set Paths:**

    -   Modify the script to define the following paths within the provided variables:
        -   `von_luschan_scale_path`: Path to a pre-defined pickle file containing the von Luschan scale data (likely requires separate creation based on the original paper's methodology).
        -   `von_luschan_scale_flat_path`: Path to a flattened version of the von Luschan scale data (optional, depending on script implementation).
        -   `img_directory`: Path to the directory containing your images.
        -   `csv_directory`: Path to the directory where you want to save the output CSV file.
4.  **Run the Script:**

    -   Execute the script using Python (`python script_name.py`).

**Output:**

The script will generate a CSV file named `von_luschan_scale_index.csv` in the specified `csv_directory`. This file will contain the following columns:

-   `filename`: Name of the analyzed image file.
-   `von_luschan_scale_index`: Index corresponding to the image's skin tone within the von Luschan scale (interpret with caution due to limitations).
-   `fitzpatrick_scale_index`: Mapped index from the von Luschan scale to the Fitzpatrick scale (another skin tone classification system, also limited).
-   `median_color`: Median color values (RGB) of the image.
