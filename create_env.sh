# Создаем файл .env
cat > .env << EOF
MONITORED_DIR=/pdf
UID=$(id -u)
GID=$(id -g)
EOF