# GrammarAI

# Installation
In order to install GrammarAI, the repository has to be cloned to a desired directory:
```
git clone https://github.com/must1d/GrammarAI.git
```
From the repository, a virtual environment has to be created and activated.
```
cd GrammarAI
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
Install all required packages:
```
pip install -r requirements.txt
```
Lastly, install a pytorch and cuda version that fits your system from:
https://pytorch.org/


# Usage
To start GrammarAI, first source the environment.
```
cd path/to/GrammarAI
source venv/bin/activate
```
From here, run the `main.py` script with the sentence to be corrected as argument:
```
python main.py "SENTENCE"
```