import pandas as pd
from AlphaProQA import AlphaProQA
from datasets import load_dataset

# Use your API keys here
coAnswer = cohere.ClientV2('CO_API_KEY_1')
coRewrite = cohere.ClientV2('CO_API_KEY_2')

# Load the QA training dataset
qaDataset = load_dataset("cardiffnlp/databench", "semeval", split="train")
print(qaDataset.column_names)

ques = qaDataset['question']
datasets = qaDataset['dataset']
expAns = qaDataset['answer']

datasetDict = {
  dName : pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dName}/all.parquet")
  for dName in set(datasets)
}

# Resutls dictionary
results = {
  'Question':[],
  'Rewriten Question': [],
  'Code':[],
  'Expected Answer Type':[],
  'Answer':[],
  'Expected Answer':[]
}

# Creating instance of AlphaProQA class
qa = AlphaProQA()

# Final for loop for running for desired rows and generating output
lastIndex = -1  # Change for each iteration if needed (last printed index of for loop)

# Fill your desired indices here
start = 0
end = len(ques)
# for i in range(len(ques)):
for i in range(start, end):
  res = qa.getAnswer(
    cohereChatAnswer=coAnswer,
    cohereChatRewrite=coRewrite,
    question=ques[i],
    datasetMetaData={
        'dataset': datasetDict[datasets[i]],
        'dataset_name':datasets[i],
      }
  )

  results['Question'].append(ques[i])
  results['Rewriten Question'].append(res['rewritten_question'])
  results['Code'].append(res['code'])
  results['Expected Answer Type'].append(res['answer_type'])
  results['Answer'].append(res['output'])
  results['Expected Answer'].append(expAns[i])

  print(f'Finished row {i}')

# Saving the results
datasetNames = {
  'Dataset Used': [qaDataset['dataset'][index] for index in range(len(results['Question']))]
}
resultTable = pd.DataFrame({**datasetNames, **results})

print()
print(resultTable)
print()

resultTable.to_csv('../results/training_data_results.csv', mode='w', index=False)
