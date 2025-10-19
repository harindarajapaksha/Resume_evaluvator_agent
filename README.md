# Resume evaluator agent

### Example usage


#### Input file formats

* Resume file format must be in file_name.txt format
* Position description must be in file_name.txt format

#### OPENAI API key

Export the API key before running this. 

Example:
```{bash}
export OPENAI_API_KEY="Your OpenAI API key"
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
