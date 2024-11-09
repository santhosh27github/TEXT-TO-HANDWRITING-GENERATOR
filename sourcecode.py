from PIL import Image, ImageDraw, ImageFont
import os
import tensorflow as tf
import numpy as np
# import fontforge

def generate_handwriting(text, font_path, font_size, output_path):
    """
    Generates an image with the given text written in a specified font and saves it to the output path.

    Parameters:
    text (str): The text to be written on the image.
    font_path (str): The file path to the .ttf font file to be used.
    font_size (int): The size of the font to be used.
    output_path (str): The file path where the generated image will be saved.

    Returns:
    None
    """
    try:
        # Create a blank image with a white background
        img = Image.new('RGB', (1000, 1000), color=(255, 255, 255))
        d = ImageDraw.Draw(img)

        # Load the font  
        font = ImageFont.truetype(font_path, font_size)

        # Write the text on the image
        d.text((50, 50), text, fill=(0, 0, 0), font=font)

        # Save the image
        img.save(output_path)
        
        print(f"Image saved to {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")


def generate_glyphs(text, char_to_num):
    """
    Generates a TTF font file with glyphs for the specified text characters, mapping each character to a unique glyph number.

    Parameters:
    text (str): The text containing characters for which glyphs will be created.
    char_to_num (dict): A dictionary mapping each character in the text to a unique glyph number.

    Returns:
    None

    Raises:
    Exception: If an error occurs during the font creation or saving process.
    """
    # Create a new font
    # font = fontforge.font()
    # font.fontname = "Generated Font"
    # font.familyname = "Generated Font Family"
    # font.fullname = "Generated Font Full Name"

    # Add glyphs to the font
    # for char in text:
    #     glyph = font.createChar(char_to_num[char], char)
    #     glyph.width = 1000

    # Save the font to a TTF file
    # font.generate("generated_font.ttf")

def generate_font(model, image, unique_characters, char_to_num):
    """
    Generates a TTF font file using a trained model to predict text from an image.

    Parameters:
    model: The trained model used to predict the text from the image.
    image: The input image from which the text is to be predicted.
    unique_characters (list): A list of unique characters that the model can predict.
    char_to_num (dict): A dictionary mapping each character to a unique glyph number for font generation.

    Returns:
    None

    Raises:
    Exception: If an error occurs during the prediction or font generation process.
    """
    # Use the trained model to predict the text
    prediction = model.predict(image)
    # Convert the prediction to a string
    text = ''.join([unique_characters[np.argmax(pred)] for pred in prediction])
    # Generate the font using the predicted text
    generate_glyphs(text, char_to_num)


def main():
    """
    Main function to handle two operations:
    1. Generating an image with handwritten text using a selected font.
    2. Recognizing handwritten text from an image and generating a custom font based on the recognized characters.

    Notes:
    - The function uses TensorFlow for image preprocessing and model definition.
    - FontForge's Python API is used for generating custom font files.
    - This is an illustrative example; the actual model should be pre-trained and loaded instead of being trained with dummy data.

    Returns:
    None

    Raises:
    Exception: If any error occurs during font selection, image preprocessing, or font generation.

    """
    choice = int(input("Enter 1 for font image generation or 2 for handwriting recognition and font generation: "))
    
    if choice == 1:
        # these are the list of font files present
        list_font_files = os.listdir(os.path.join(os.getcwd(), "fonts"))

        # Display available fonts to the user
        print("Available fonts:")
        for i, font in enumerate(list_font_files, 1):
            print(f"{i}. {font}")

        # Get user input for the font selection
        font_choice = int(input("Enter the number of the font you want to use: "))
        if font_choice < 1 or font_choice > len(list_font_files):
            print("Invalid font selection.")
            return

        # Get user input for the text
        text = input("Enter the text you want to generate: ")


        # Define font path based on user selection
        font_path = os.path.join(os.getcwd(), "fonts", list_font_files[font_choice - 1])
        font_size = 50
        output_path = os.path.join(os.getcwd(), "output_handwritten_image.png")

        # Generate the handwritten text image
        generate_handwriting(text, font_path, font_size, output_path)

    elif choice == 2:
        # Load the handwriting image
        image_path = 'handwritten_text.png'
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1)

        # Preprocess the image
        image = tf.image.resize(image, (200, 50))
        image = tf.cast(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  # Add batch dimension, shape: (1, 200, 50, 1)

        # Extract features from the image
        features = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image)
        features = tf.keras.layers.MaxPooling2D((2, 2))(features)
        flattened_features = tf.keras.layers.Flatten()(features)

        # Reshape features for LSTM layer
        batch_size, flat_feature_size = flattened_features.shape
        timesteps = 1  # Example: treating the entire flattened feature as a single timestep
        features_per_timestep = flat_feature_size
        reshaped_features = tf.reshape(flattened_features, (batch_size, timesteps, features_per_timestep))

        # Define the unique characters (Example)
        unique_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        char_to_num = {char: i for i, char in enumerate(unique_characters)}

        # Define the machine learning model
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(reshaped_features.shape[1], reshaped_features.shape[2])),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(unique_characters), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Dummy labels for illustration (normally you would use real labels)
        labels = np.random.randint(len(unique_characters), size=(batch_size,))
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(unique_characters))

        # Train the model (Dummy example)
        model.fit(reshaped_features, labels, epochs=150, batch_size=8)

        # Generate the font from the user's handwriting image
        generate_font(model, reshaped_features, unique_characters, char_to_num)
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()