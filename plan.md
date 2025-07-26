1. Reorganize Schemas

  Move core domain models to jesktop/models/ with separate files:
- jesktop/models/note.py
- jesktop/models/image.py
- jesktop/models/relationships.py

2. Fix Image Resolution

- Create a configurable PathResolver class
- Define attachment folder patterns in config
- Add proper error handling instead of silent fallbacks

3. Improve Image Storage

- Add composite index: (note_id, relative_path) -> image_id
- Use deterministic image IDs based on note_id + path
- Cache path lookups

4. Standardize Reference Processing

- Create unified ReferenceParser class
- Handle all reference types (wikilinks, images, excalidraw) consistently
- Move regex patterns to config

5. Frontend Robustness

- Replace manual regex with a proper markdown parser
- Add error states for missing images
- Implement retry logic for failed image loads
