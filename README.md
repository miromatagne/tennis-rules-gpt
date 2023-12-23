# Chat-ATP

Answer custom user questions about [ATP (Association of Tennis Professionals) rules](https://www.atptour.com/en/corporate/rulebook).

The official rules are stored locally in the `data` folder in several PDF files. Along with the PDF files, a metadata JSON file contains metadata about the overall data (date of retrieval, language, ...) but also about each individual file such as the title or url.

In the `data_extractor.py` script, the text content from the PDF files is extracted and stored into a Python dictionary.

