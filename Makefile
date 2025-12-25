help:
	@echo "Available commands:"
	@echo "  make run    - Run app locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

run:
	python app.py

docker-build:
	docker build -t sentiment-app:latest .

docker-run:
	docker run --rm -it -p 5000:5000 sentiment-app:latest
