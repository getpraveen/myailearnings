Steps:
1.Install python 3.11.9, Install VSCode
2.Create a folder on desktop, then open it from VSCode  File --> Open Folder
3.Create .venv file (virtual environment file, its name can be any thing, here I have named it .venv)
    -- Python -m venv .venv
4.Activate venv following below.

https://stackoverflow.com/questions/4037939/powershell-says-execution-of-scripts-is-disabled-on-this-system

To activate venv in windows, first do this.
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

To revert above setting [Ignore this, while you are working on VSCode]
Set-ExecutionPolicy Restricted

Run this --> .\.venv\Scripts\Activate.ps1 
5.pip install poetry
6.poetry init, check that if this setting exists in pyproject.toml - python = ">=3.11,<3.13", edit it if required.
7.poetry add ollama chromadb pdfplumber langchain langchain-core langchain-ollama langchain-community langchain_text_splitters unstructured unstructured[all-docs] fastembed pdfplumber sentence-transformers elevenlabs
8.pip install ipykernel
9.sudo apt-get update
10.sudo apt-get install -y poppler-utils libpoppler-cpp-dev
11.sudo apt-get install tesseract-ocr
