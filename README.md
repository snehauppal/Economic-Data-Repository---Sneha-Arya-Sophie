# Economic-Data-Repository---Sneha-Arya-Sophie





## PART THREE EXECUTION INSTRUCTIONS

This section explains how to run the FastAPI application defined in
`api/app.py`, which exposes a Qwen-based Retrieval-Augmented Generation (RAG)
API over the cleaned economic dataset.

---

## 1. Clone the Repository

Clone the repository to the VM or local machine:

```bash
git clone https://github.com/aryakumar2005-rgb/Economic-Data-Repository---Sneha-Arya-Sophie.git
cd Economic-Data-Repository---Sneha-Arya-Sophie


## SCP Command to get dataset into virtual machine - first download the dataset from the data folder in our repo and then use this command (altered for your VM to get the dataset from your local downloads into your VM

scp -i ~/.ssh/ds2002/ds2002_shared_student_key \
    ~/Downloads/etl_cleaned_dataset.csv \
    student@34.48.211.118:/home/student/Economic-Data-Repository---Sneha-Arya-Sophie/data/

### VM Environment Setup - The API is designed to run on a Linux VM. Enter your virtual machine and enter this
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-venv python3-pip git build-essential cmake ninja-build libopenblas-dev

mkdir -p ~/hf-cli-bot
cd ~/hf-cli-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
pip install huggingface_hub fastapi uvicorn pandas

##activate the virtual environment and navigate to api directory  
source ~/hf-cli-bot/.venv/bin/activate
cd ~/Economic-Data-Repository---Sneha-Arya-Sophie/api

##start the server (server commmand and curl example is also included as a comment in the repo app.py)
python -m uvicorn app:app --host 0.0.0.0 --port 8000

##example curl - Once the FastAPI server is running, the API can be queried directly from the
command line using curl

curl -X POST http://34.48.211.118:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What was the unemployment rate in the United States in 2019?"
      } 
{
  "answer": "The unemployment rate in the United States in 2019 was 3.7%.",
  "sources": [
    "Country: United States, Year: 2019, Unemployment rates: 3.7, ..."
  ]
}

  




