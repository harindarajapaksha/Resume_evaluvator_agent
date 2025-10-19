# Resume evaluator agent

### Example usage


#### Input file formats

* Resume file format must be in file_name.txt format
* Position description must be in file_name.txt format

#### .env file

This file is not included in the repo. Need to create this file and include the following line with the API keay. 

Example:
```{bash}
OPENAI_API_KEY="API-KEY"
```

#### Commandline usage

```{bash}
usage: main.py [-h] -r RESUME -p POSITION

Resume assessor

options:
  -h, --help            show this help message and exit
  -r RESUME, --resume RESUME
                        Path to the resume.txt file
  -p POSITION, --position POSITION
                        Path to the position_description.txt file
```
