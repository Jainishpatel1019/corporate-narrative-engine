.PHONY: install test ingest index demo

install:
	pip install -r requirements.txt

test:
	pytest tests/

ingest:
	@echo "Ingestion not implemented yet"

index:
	@echo "Indexing not implemented yet"

demo:
	@echo "Demo not implemented yet"
