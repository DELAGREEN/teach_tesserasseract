git clone --recurse-submodules https://github.com/DELAGREEN/teach_tesserasseract.git
cd teach_tesserasseract/tesstrain
make tesseract-langdata
pip install -r requirements.txt