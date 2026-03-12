from pathlib import Path

project_name = "vehicle_insurance"

list_of_files = [

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",  
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    f"src/{project_name}/configuration/__init__.py",
    f"src/{project_name}/configuration/mongo_db_connection.py",
    f"src/{project_name}/configuration/aws_connection.py",
    f"src/{project_name}/cloud_storage/__init__.py",
    f"src/{project_name}/cloud_storage/aws_storage.py",
    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/proj1_data.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",
    f"src/{project_name}/entity/estimator.py",
    f"src/{project_name}/entity/s3_estimator.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "pyproject.toml",
    "config/model.yaml",
    "config/schema.yaml",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.touch(exist_ok=True)