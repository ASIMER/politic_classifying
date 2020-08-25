import pandas as pd
version = 19
textes = pd.read_csv("data\primary_debates.csv")

print(textes.describe())
#data filter part ========================================================================================
#text filter part templates
junk_axis = ['Line', 'Speaker', 'Date', 'Location', 'URL']
candidates = ['Bush', 'Carson', 'Chafee', 'Christie', 'Clinton', 'Cruz', 'Fiorina', 'Gilmore', 'Graham', 'Huckabee', 'Jindal', 'Kasich', "O'Malley", 'Pataki', 'Paul', 'Perry', 'Rubio', 'Sanders', 'Santorum', 'Trump', 'Walker', 'Webb']
parties = ['Republican', 'Democratic']
#candidates_textes = textes[textes["Speaker"].isin(candidates)]

#drop dublicates
textes = textes.drop_duplicates()

#filter parties
textes = textes[textes.Party.isin(parties)]

#filter candidates
textes = textes[textes.Speaker.isin(candidates)]
#drop junk columns
for col in junk_axis:
    textes = textes.drop(col, axis=1)

#unify classes "Republican" ->1   "Republican Undercard"->1   "Democratic"->0
textes = textes.replace(['Republican', 'Republican Undercard', 'Democratic'],[1, 1, 0])
#lower class names Text ->text  Party->class
textes = textes.set_axis(['text', 'class'], axis='columns')
textes = textes.sort_values(by=['class'])
#new index numeration
textes = textes.set_axis(pd.RangeIndex(stop=textes.shape[0]), axis='index')

#filter short textes
for i in range(len(textes['text'])):
    if len(textes['text'][i])<=200:
        textes = textes.drop(i)
print(textes['class'].describe())

#new index numeration
textes = textes.set_axis(pd.RangeIndex(stop=textes.shape[0]), axis='index')

#filter short texts from republicans
for i in range(len(textes['text'])):
    if len(textes['text'])<=i:
        break
    if len(textes['text'][i])<=600:
        if textes.values[i][1] == 1:
            textes = textes.drop(i)
print(textes['class'].describe())
#new index numeration
textes = textes.set_axis(pd.RangeIndex(stop=textes.shape[0]), axis='index')
print(textes['class'].tail())
print(textes['class'][2])
print(textes.text[1921])
print(textes.text[1922])
print(textes.text[1923])
print(textes.text[1924])

textes = textes.truncate(after=1919)
print(textes['class'].describe())

#random shuffle
textes = textes.sample(frac=1)

#lower all words
textes['text'] = textes['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#remove punctuation
#textes['text'] = textes['text'].str.replace('[^\w\s]',' ')

print("0", textes['class'].to_list().count(0))
print("1", textes['class'].to_list().count(1))
#data filter part end ====================================================================================
print(textes.shape)
textes.to_csv("data/cleaned_textes_v{}.csv".format(version), index=False)