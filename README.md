# Image_Denoiser
# Image Denoising Using Autoencoder

This project focuses on image denoising using an autoencoder model. The workflow includes splitting the dataset, processing images, training the model, evaluating performance, and predicting denoised images.

## Project Structure

- high and low directories: Contain high and low-resolution images.
- train and test directories: Split into high and low subdirectories for training and testing datasets.
- train_patches and test_patches directories: Contain patches of images used for training and testing.
- predicted directory: Stores the denoised images predicted by the model.

## Workflow

1. *Data Preparation*
    - Split images into training and testing sets.
    - Resize images and create patches.

2. *Visualization*
    - Display pairs of training and testing images.
    - Plot pixel value distributions.
    - Calculate and display PSNR and SSIM metrics.

3. *Model Training*
    - Define an autoencoder model.
    - Train the model on training image patches.
    - Evaluate the model on training and testing datasets.

4. *Evaluation*
    - Calculate PSNR for the test dataset.

5. *Prediction*
    - Predict and save denoised images for the test dataset.

## Usage

### Data Preparation

1. Split the dataset into training and testing sets:
    python
    # Define paths
    high_path = '/content/drive/MyDrive/high'
    low_path = '/content/drive/MyDrive/low'
    train_high_path = 'train/high'
    train_low_path = 'train/low'
    test_high_path = 'test/high'
    test_low_path = 'test/low'

    # Create directories for train and test sets
    os.makedirs(train_high_path, exist_ok=True)
    os.makedirs(train_low_path, exist_ok=True)
    os.makedirs(test_high_path, exist_ok=True)
    os.makedirs(test_low_path, exist_ok=True)

    # Function to move files to the appropriate directory
    def move_files(src_dir, dest_dir, file_list):
        for file_name in file_list:
            src_file = os.path.join(src_dir, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            shutil.move(src_file, dest_file)

    # Get list of files in high and low directories
    high_files = sorted(os.listdir(high_path))
    low_files = sorted(os.listdir(low_path))

    # Split files into train and test sets
    test_high_files = high_files[:80]
    train_high_files = high_files[80:]
    test_low_files = low_files[:80]
    train_low_files = low_files[80:]

    # Move files to test directories
    move_files(high_path, test_high_path, test_high_files)
    move_files(low_path, test_low_path, test_low_files)

    # Move files to train directories
    move_files(high_path, train_high_path, train_high_files)
    move_files(low_path, train_low_path, train_low_files)
    

2. Process and save patches:
    python
    def process_and_save_patches(src_path, dest_path, img_size=(1024, 1024), patch_size=(256, 256)):
        image_files = sorted(os.listdir(src_path))

        for img_file in image_files:
            img_path = os.path.join(src_path, img_file)
            image = Image.open(img_path).convert('RGB')
            resized_image = resize_image(image, img_size)
            patches = create_patches(resized_image, patch_size)

            base_name = os.path.splitext(img_file)[0]
            for idx, patch in enumerate(patches):
                patch_name = f"{base_name}_patch_{idx}.png"
                patch_path = os.path.join(dest_path, patch_name)
                patch.save(patch_path)

    # Process and save patches for train and test datasets
    process_and_save_patches(train_high_path, train_high_patches_path)
    process_and_save_patches(train_low_path, train_low_patches_path)
    process_and_save_patches(test_high_path, test_high_patches_path)
    process_and_save_patches(test_low_path, test_low_patches_path)
    

### Model Training and Evaluation

1. Train the autoencoder model:
    python
    # Define and compile autoencoder model
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    # Train the model
    history = autoencoder.fit(train_data_loader,
                              epochs=epochs,
                              steps_per_epoch=len(train_data_loader),
                              validation_data=test_data_loader,
                              validation_steps=len(test_data_loader),
                              verbose=1)
    

2. Evaluate PSNR on test dataset:
    python
    # Evaluate PSNR
    test_psnr = evaluate_psnr(autoencoder, test_data_loader)
    print(f"Test PSNR: {test_psnr:.2f} dB")
    

### Prediction

1. Predict and save denoised images:
    python
    # Predict denoised images
    for img_name in os.listdir(test_low_dir):
        img_path = os.path.join(test_low_dir, img_name)

        # Load and preprocess the test image
        original_img, img_array = load_image(img_path)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict denoised image using the model
        denoised_img = autoencoder.predict(img_array)
        denoised_img = denoised_img.squeeze()

        # Save the denoised image
        save_image(denoised_img, img_name)

    print(f"Predicted images saved to {predicted_dir}")
    

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pillow
- Matplotlib
- scikit-image

Install the required packages using:
bash
pip install tensorflow numpy pillow matplotlib scikit-image


## Results

- Display and compare original and denoised patches.
- Visualize PSNR 29.92 db
- Predict and save denoised images.

## Conclusion

This project demonstrates the use of an autoencoder for image denoising, including data preparation, model training, evaluation, and prediction of denoised images. The evaluation metrics and visualizations help in understanding the model's performance.

For further details, refer to the code provided in the project files.
