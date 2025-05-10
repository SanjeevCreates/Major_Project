# main.py
from src.feature_selection import wolf_optimizer
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.app_gradio import launch_gradio_app

def main():
    # Feature selection
    selected_features = wolf_optimizer()
    print(f"Selected Features: {selected_features}")

    # Train model
    model = train_model(selected_features)

    # Evaluate model
    accuracy = evaluate_model(model)
    print(f"Model Accuracy: {accuracy}")

    # Launch Gradio app
    launch_gradio_app()

if __name__ == "__main__":
    main()


