#### Установка Tesseract <br>
    --- AstraLinux ---
        ./autogen.sh
        ./configure LDFLAGS="-lstdc++fs"   # это решает ошибку с filesystem
        make training
        sudo make install LIBS="-lstdc++fs"
        sudo make training-install LIBS="-lstdc++fs"
        sudo ldconfig

#### Другие ОС
        



