from nerfbaselines import register

register({
    "method_class": "my_method:MyMethod",
    "conda": {
        "environment_name": "my_method",
        "python_version": "3.11",
        "install_script": """
# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 'numpy<2.0.0' \
    --index-url https://download.pytorch.org/whl/cu118
""",
    },
    "id": "my-method",
    "metadata": {},
})
