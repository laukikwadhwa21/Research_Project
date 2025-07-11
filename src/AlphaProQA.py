import pandas as pd
import cohere

class AlphaProQA:
  '''Class for the approach in the Colab notebook: /notebooks/AlphaProQA.ipynb. For getting the answer to a question, create two model instances (here, using cohere) and pass the required data to the getAnswer() function.
  For specific usage and detailed description of code approach, refer the notebook mentioned.'''
  def __init__(self):
    self.__rewrite_prompt = """You will be provided with two pieces of information. The first being a question and the second being the column names along with data types of a dataset. Your objective is twofold, the first to predict the datatype of the answer and second to paraphrase the question aptly such that the next person could generate the python code to required to answer the question while keeping the answer type the same as the given question. You are provided a two examples below.
Remember to not change what the original question is actually asking.

Few Shot Examples:
Question: Is the person with the highest net worth self-made?
Dataset Name: 001_Forbes
Dataset Table Schema: selfMade (bool), finalWorth (int64), city (string), title (string), gender (string), age (float64), rank (int64), philanthropyScore (float64), category (string), source (string), country (string)
Answer Type: bool
Paraphrased Question: Does the billionaire with the maximum final worth have self made attribute set to True?

Question: Did any children below the age of 18 survive?
Dataset Name: 002_Titanic
Dataset Table Schema: Age (float64), Siblings_Spouses Aboard (int64), Sex (string), Name (string), Pclass (int64), Fare (float64), Survived (bool)
Answer Type: bool
Paraphrased Question: Were there any survivors aged under 18?

Instruction for you to perform:
"""

    self.__codegen_prompt = '''You will be provided four pieces of information all of which are provided in the means of strings.
1. Dataset name:
2. Dataset Table Schema:
3. Question:
4. Expected Answer Type:

Your objective is to create a python code to answer the question given the dataset schema. Here is the function you will be needing to complete:
def answer_question(db:, datasetTableSchema, question, expectedAnswerType):
	answer = (Here you generate the code which is needed to find the answer)
	return answer

Assume that the pandas library has been imported as pd.
Your answer should only contain the function definition. Assume that the dataset schema (containing column names and their datatypes in paranthesis) given is correct. The generated code should be correct. Do not attempt to change the dataset.
Your final answer data type should be one of the following categories:
1. Boolean: One of True or False.
2. Category: A string. For example - CEO, hello, drugstores.
3. Number: A numerical value. For example - 20, 23.3223, 414901.0.
4. list[category]: A list of strings. For example - ['India', 'Japan', 'China'], ['Ram', 'Shyam', 'Mohan']. Here, each entry should be enclosed within single quotes.
5. list[number]: A list of numbers. For example - [20.0, 30.4, 42.1], [171000, 129000, 111000, 107000, 106000, 91400].
When the question requests more than value, the expected answer type might be a list of strings or numbers. Ensure that lists are enclosed within square brackets.

Few Shot Examples:
Example 1:
1. Dataset name: 001_Forbes
2. Dataset Table Schema: selfMade (bool), finalWorth (int64), city (string), title (string), gender (string), age (float64), rank (int64), philanthropyScore (float64), category (string), source (string), country (string)
3. Question: Does the individual with the highest final worth value have the selfMade attribute set to True?
4. Expected Answer Type: bool

Answer:
def answer_question(dataset, datasetTableSchema, question, expectedAnswerType):
	max_worth_individual = dataset.loc[dataset["finalWorth"] == dataset["finalWorth"].max()]
	is_self_made = max_worth_individual["selfMade"].bool()

	return is_self_made

Now, complete the following:'''

  # Rewriting the question and predicting answer data type
  def __processQuestion(self, cohereChat:cohere.ClientV2, question:str, dataset_name:str, schema:str) -> str:
    '''
    Process the question and return predicted answer type and paraphrased question, in that order.
    Parameter 'schema' is a comma separated list of strings - column name (column data type).
    '''
    # Prepare prompt
    prompt = self.__rewrite_prompt + \
    f'''Question: {question}
Dataset: {dataset_name}
Dataset Table Schema: {schema}
'''
    # Generate response using Cohere API
    response = cohereChat.chat(
          model="command-r-plus-08-2024",
          messages=[{"role": "user", "content": prompt}],
      )
    # Extract answer type and paraphrased question
    answer_type = response.message.content[0].text.split("\n")[0][0]
    if(answer_type[0] == 'A'):
      answer_type = response.message.content[0].text.split("\n")[0][13:]
      paraphrased_question = response.message.content[0].text.split("\n")[1][22:]
    else:
      answer_type = response.message.content[0].text.split("\n")[1][13:]
      paraphrased_question = response.message.content[0].text.split("\n")[0][22:]

    return answer_type, paraphrased_question

  # Generating the code
  def __generateCode(self, cohereChat:cohere.ClientV2, question:str, metaData:dict) -> str:
    '''
    Generate code string for answering the paraphrased question.
    Parameter 'metaData' dictionary:
      'dataset_name': str,
      'columns': list[str],
      'answer_type': str
    '''
    codeResponse = cohereChat.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": self.__codegen_prompt + f'''
1. Dataset name: {metaData['dataset_name']}
2. Dataset Table Schema: {str(metaData['columns'])[1:-1]}
3. Question: {question}
4. Expected Answer Type: {metaData['answer_type']}'''
                  }])

    text = codeResponse.message.content[0].text
    text = text.strip("```").lstrip("python\n")
    return text

  # Extracting code from the code-string for running
  def __extractFunctionFromString(self, function_str:str):
    '''
    Take a string containing a function named 'answer_question' and return the function in scope.
    The function will have access to local and global variables.
      '''
    namespace = {**globals(), **locals()}
    exec(function_str, namespace)
    return namespace['answer_question']  # 'answer_question' is the default name in the code string
  
  # Getting the dataset schema
  def __getDatasetSchema(self, df:pd.DataFrame) -> list[str]:
    '''
    Get the dataset schema from the pandas.DataFrame object.
    List entry is - column name (column data type)
    '''
    schema = df.dtypes
    schema_string = ""
    for col, dtype in schema.items():
      if dtype == "bool":
          dtype_name = "bool"
      elif dtype == "int64":
            dtype_name = "int64"
      elif dtype == "double":
          dtype_name = "float64"
      elif dtype == "object":
          dtype_name = "string"
      else:
          dtype_name = dtype.name

      schema_string += f"{col} ({dtype_name}), "
    # Remove the trailing comma and space
    schema_string = schema_string.rstrip(", ")
    return schema_string.split(", ")

  # Final function to get answer
  def getAnswer(self, cohereChatAnswer:cohere.ClientV2, cohereChatRewrite:cohere.ClientV2, question:str, datasetMetaData:dict) -> str:
    '''
    Get final answer along with state information given a question in natural language and a dataset.
    Parameter 'datasetMetaData' dictionary:
      'dataset': pandas.DataFrame,
      'dataset_name': str

    Output dictionary:
      'original_question' : str,
      'rewritten_question' : str,
      'code' : str,  (code string)
      'answer_type' : str,  (predicted answer type in a string)
      'output' : Any  (actual answer - data type of this object depends on the code generated and the question, but the data type of the entity this represents is indicated in 'answer_type' entry)
    '''
    newQuestion = ''
    codeString = ''
    answerType = None
    output = '-'
    try:
      columns = self.__getDatasetSchema(datasetMetaData['dataset'])
      datasetMetaData['columns'] = columns

      answerType, newQuestion = self.__processQuestion(cohereChatRewrite, question, datasetMetaData['dataset_name'], str(columns)[1:-1])
      datasetMetaData['answer_type'] = answerType

      codeString = self.__generateCode(cohereChatAnswer, newQuestion, datasetMetaData)

      function = self.__extractFunctionFromString(codeString)

      output = function(datasetMetaData['dataset'], columns, newQuestion, datasetMetaData['answer_type'])
      del function
    except Exception as e:
      print(f"Error: {e}\nQuestion: {question}")
      output = '-'

    return {
        'original_question' : question,
        'rewritten_question' : newQuestion,
        'code' : codeString,
        'answer_type' : answerType,
        'output' : output
    }
