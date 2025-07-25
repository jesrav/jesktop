# Autoload environment variables from .env file
set dotenv-load:= true

install:
    uv sync --all-groups
    uv run pre-commit install

lint:
    uv run pre-commit run --all-files

test:
    uv run pytest
    
run:
    uv run fastapi dev app.py

build_linux:
    docker build --platform linux/amd64 -t note_chat_img -f Dockerfile .

build:
    docker build -t note_chat_img -f Dockerfile .
    
deploy: build_linux
    docker tag note_chat_img registry.digitalocean.com/jesravnbol/note_chat_img
    docker push registry.digitalocean.com/jesravnbol/note_chat_img

run_in_docker:
	docker run -p 8000:8000 --env-file=.env note_chat_img

ingest:
    uv run scripts/ingest.py \
    --in-folder data/notes \
    --outfile-vector-db data/vector.json
