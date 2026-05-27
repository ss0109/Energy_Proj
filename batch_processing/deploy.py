from batch_processing.prefect_flow import energy_pipeline

# Run from the project root with:
#   python -m batch_processing.deploy
if __name__ == "__main__":
    energy_pipeline.serve(name="energy-pipeline")
