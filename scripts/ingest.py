"""CLI for getting embeddings for sample data and saving to either local Vector DB or a DBX Mosaic Vector DB"""

import argparse
from pathlib import Path

from jesktop.config import settings
from jesktop.embedders.voyage_embedder import VoyageEmbedder
from jesktop.image_store.local import LocalImageStore
from jesktop.ingestion.orchestrator import IngestionOrchestrator
from jesktop.vector_dbs.local_db import LocalVectorDB


def main(
    in_folder: str,
    local_outfile_vector_db: str,
    local_outfile_image_store: str,
) -> None:
    # Setup paths and services
    folder = Path(in_folder)
    vector_db_output = Path(local_outfile_vector_db)
    image_store_output = Path(local_outfile_image_store)
    embedder = VoyageEmbedder(api_key=settings.voyage_ai_api_key)
    vector_db = LocalVectorDB(filepath=vector_db_output)
    image_store = LocalImageStore(filepath=image_store_output)

    orchestrator = IngestionOrchestrator(
        embedder=embedder,
        vector_db=vector_db,
        image_store=image_store,
    )
    orchestrator.ingest(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-folder", type=str, required=True, help="Folder containing markdown files"
    )
    parser.add_argument(
        "--outfile-vector-db",
        type=str,
        required=False,
        help="Local output vector db file",
        default=settings.local_vector_db_path,
    )
    parser.add_argument(
        "--outfile-image-store",
        type=str,
        required=False,
        help="Local output image store file",
        default=settings.local_image_store_path,
    )

    args = parser.parse_args()

    main(
        in_folder=args.in_folder,
        local_outfile_vector_db=args.outfile_vector_db,
        local_outfile_image_store=args.outfile_image_store,
    )
