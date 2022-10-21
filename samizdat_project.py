import pandas as pd
from my_functions import marc_parser_1_field, unique_elem_from_column_split, cSplit, replacenth, gsheet_to_df, df_to_gsheet, get_cosine_result
import regex as re
from functools import reduce
import numpy as np
import copy
import requests
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError
from http.client import RemoteDisconnected
import time
import xml.etree.ElementTree as et
from google_drive_research_folders import cr_projects
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import gspread as gs
from tqdm import tqdm
import datetime
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import copy
import json
import difflib
from qwikidata.sparql  import return_sparql_query_results
import requests
import ast

#%% def

def marc_parser_dict_for_field(string, subfield_code):
    subfield_list = re.findall(f'{subfield_code}.', string)
    dictionary_field = {}
    for subfield in subfield_list:
        subfield_escape = re.escape(subfield)
        string = re.sub(f'({subfield_escape})', r'❦\1', string)
    for subfield in subfield_list:
        subfield_escape = re.escape(subfield)
        regex = f'(^)(.*?\❦{subfield_escape}|)(.*?)(\,{{0,1}})((\❦{subfield_code})(.*)|$)'
        value = re.sub(regex, r'\3', string).strip()
        dictionary_field[subfield] = value
    return dictionary_field

def catch(e):
    try:
        return '❦'.join([f"{a['text']}|{len(a['sources']['s']) if isinstance(a['sources']['s'], list) else 1}" for a in e['record']['recordData']['mainHeadings']['data']])
    except (TypeError, KeyError):
        if isinstance(e['record']['recordData']['mainHeadings']['data']['sources']['s'], list):
            return f"{e['record']['recordData']['mainHeadings']['data']['text']}|{len(e['record']['recordData']['mainHeadings']['data']['sources']['s'])}"
        else:
            return f"{e['record']['recordData']['mainHeadings']['data']['text']}|1"

def create_worksheet(sheet, worksheet_name, df):    
    try:
        set_with_dataframe(sheet.worksheet(worksheet_name), df)
    except gs.WorksheetNotFound:
        sheet.add_worksheet(title=worksheet_name, rows="100", cols="20")
        set_with_dataframe(sheet.worksheet(worksheet_name), df)
        
def harvest_viaf_by_titles(df, column_id, data_scope_name, subfield_code, df_with_titles, field_for_title, title_subfield):
    df['odpytanie po tytułach'] = np.nan
    df['odpytanie po tytułach'] = df['odpytanie po tytułach'].astype(object)

    df_grouped = df.groupby(column_id)
    
    new_df = pd.DataFrame()
    for name, group in tqdm(df_grouped, total=len(df_grouped)):
        testy = []
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            locations = row['name_form_id'].split('|')
            list_of_suggested_viafs = []
            for location in locations:
                location = location.split('-')
                if location[0] == data_scope_name:
                    title = df_with_titles[df_with_titles['id'] == int(location[1])][field_for_title].reset_index(drop=True)[0]
                    title = marc_parser_dict_for_field(title, subfield_code)[title_subfield]
                    try:
                        url = f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&httpAccept=application/json"
                        response = requests.get(url)
                        response.encoding = 'UTF-8'
                        samizdat_json = response.json()
                        list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
                        samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                        for number in list_of_numbers_of_records:
                            url = re.sub('\s+', '%20', f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                            response = requests.get(url)
                            response.encoding = 'UTF-8'
                            samizdat_json_next = response.json()
                            samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                            samizdat_json += samizdat_json_next 
                    except (ValueError, KeyError):
                        try:
                            url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                            response = requests.get(url)
                            response.encoding = 'UTF-8'
                            samizdat_json = response.json()  
                            list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
                            samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                            for number in list_of_numbers_of_records:
                                url = re.sub('\s+', '%20', f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                                response = requests.get(url)
                                response.encoding = 'UTF-8'
                                samizdat_json_next = response.json()
                                samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                                samizdat_json += samizdat_json_next
                        except (ValueError, KeyError):
                            break                 
                    samizdat_personal = [[catch(e), e['record']['recordData']['viafID']] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
                    try:
                        samizdat_json_title = [[e['record']['recordData']['titles']['author']['text'], e['record']['recordData']['titles']['author']['@id'].split('|')[-1]] for e in samizdat_json]
                    except (TypeError, KeyError):
                        samizdat_json_title = []
                    
                    suggested_viafs = samizdat_personal + samizdat_json_title
                    
                try:
                    list_of_suggested_viafs.append(suggested_viafs)
                except UnboundLocalError:
                    pass
            if len(list_of_suggested_viafs) > 0:
                yyy = [[a for a in e] for e in list_of_suggested_viafs]
                yyy = [e for sub in yyy for e in sub]
                testy.append(yyy)
            testy2 = [e for s in testy for e in s]
            testy2 = [list(x) for x in set(tuple(x) for x in testy2)]
            group['odpytanie po tytułach'] = '‽'.join(['|'.join(e) for e in testy2]) 
        new_df = new_df.append(group)   
    return new_df.reset_index(drop=True)   

def get_wikidataID(x):
    try:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        return re.findall('Q\d+', x[0]['autor.value'])[0]
    except (TypeError, ValueError):
        return np.nan      

#%% date
now = datetime.datetime.now()
year = now.year
month = '{:02}'.format(now.month)
day = '{:02}'.format(now.day)

#%% google authentication & google drive
#autoryzacja do tworzenia i edycji plików
gc = gs.oauth()
#autoryzacja do penetrowania dysku
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

#%% google drive files
file_list = drive.ListFile({'q': f"'{cr_projects}' in parents and trashed=false"}).GetList() 
file_list = drive.ListFile({'q': "'1UdglvjjX4r2Hzh5BIAr8FjPuWV89NVs6' in parents and trashed=false"}).GetList()
#[print(e['title'], e['id']) for e in file_list]
nodegoat_people = [file['id'] for file in file_list if file['title'] == 'nodegoat_people_2020_12_10'][0]
nodegoat_people_sheet = gc.open_by_key(nodegoat_people)
nodegoat_people_sheet.worksheets()

nodegoat_people_df = get_as_dataframe(nodegoat_people_sheet.worksheet('Arkusz1'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()[['Project_ID', 'Index_Name', 'Other_Name_Form']]
nodegoat_people_df['Project_ID'] = nodegoat_people_df['Project_ID'].astype(int)
samizdat_people_other_names = cSplit(nodegoat_people_df[['Project_ID', 'Other_Name_Form']], 'Project_ID', 'Other_Name_Form', '; ')
samizdat_people_other_names = samizdat_people_other_names[samizdat_people_other_names['Other_Name_Form'].notnull()].rename(columns={'Other_Name_Form': 'Index_Name'})
nodegoat_people_df = pd.concat([nodegoat_people_df[['Project_ID', 'Index_Name']], samizdat_people_other_names]).sort_values('Project_ID').drop_duplicates().reset_index(drop=True)
# nodegoat_people_df = nodegoat_people_df[nodegoat_people_df['Project_ID'].isin([350, 1492, 477, 53, 2082, 13, 16, 22])]
nodegoat_people_tuples = list(nodegoat_people_df.to_records(index=False))

#%% harvestowanie viaf i wikidaty

#odpytanie po tytułach NOWY POCZĄTEK - KROK NR 1

tytuly_bn = [file['id'] for file in file_list if file['title'] == 'samizdat_kartoteka_osób'][0]
tytuly_bn_sheet = gc.open_by_key(tytuly_bn)
tytuly_bn_sheet.worksheets()

tytuly_bn_df = get_as_dataframe(tytuly_bn_sheet.worksheet('bn_books'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1)
tytuly_cz_books_df = get_as_dataframe(tytuly_bn_sheet.worksheet('cz_books'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1)

nodegoat_people_df_location = get_as_dataframe(nodegoat_people_sheet.worksheet('Arkusz1'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()[['Project_ID', 'name_form_id']]

nodegoat_people_df = nodegoat_people_df.merge(nodegoat_people_df_location, on='Project_ID', how='left')

test = copy.deepcopy(nodegoat_people_df)
# test = test[test['samizdatID'] != 53]

# test = pd.concat(e[1] for e in list(test)[11:]).groupby('Project_ID')

new_df = harvest_viaf_by_titles(test, 'Project_ID', 'bn_books', '\$', tytuly_bn_df, 200, '%a')
        
#podobieństwo nazewnictwa dla tych samych tytułów - KROK NR 2
new_df['sugestia po tytułach'] = np.nan
new_df['sugestia po tytułach'] = new_df['sugestia po tytułach'].astype(object)   
for i, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
    if row['odpytanie po tytułach'] != '':
        # i = 1
        # row = new_df.iloc[i,:]
        project_name = row['Index_Name']
        slownik = {}
        for el in row['odpytanie po tytułach'].split('‽'):
            slownik[el.split('|')[-1]] = [{f"{e}|{el.split('|')[-1]}":0} for e in el.split('|')[0].split('❦')]
        
        for k, v in slownik.items():
            for index, element in enumerate(v):
                for name, similarity in element.items():
                    only_name = name.split('|')[0]
                    slownik[k][index][name] = difflib.SequenceMatcher(a=project_name, b=only_name).ratio()
        df = [e for sub in list(slownik.values()) for e in sub]
        df = [[[k.split('|')[0], k.split('|')[-1], v] for k,v in e.items()] for e in df]
        df = [e for sub in df for e in sub]
        value = max(df, key=lambda x: x[-1])
        value = '|'.join(str(e) for e in value)
        new_df.at[i, 'sugestia po tytułach'] = value

#wyszukanie indentyfikatorów w wikidacie - KROK NR 3

url = 'https://query.wikidata.org/sparql'
new_df['wikidata'] = np.nan
new_df['wikidata'] = new_df['wikidata'].astype(object)   
        
for i, row in tqdm(new_df.iterrows(), total=new_df.shape[0]):
    while True:
        try:
            viaf = row['sugestia po tytułach'].split('|')[1]
            sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
              ?autor wdt:P214 "{viaf}" ;
              optional {{ ?autor wdt:P19 ?birthplace . }}
              optional {{ ?autor wdt:P569 ?birthdate . }}
              optional {{ ?autor wdt:P570 ?deathdate . }}
              optional {{ ?autor wdt:P20 ?deathplace . }}
              optional {{ ?autor wdt:P21 ?sex . }}
              optional {{ ?autor wdt:P106 ?occupation . }}
              optional {{ ?autor wdt:P742 ?pseudonym . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
            results = requests.get(url, params = {'format': 'json', 'query': sparql_query})
            results = results.json()
            results_df = pd.json_normalize(results['results']['bindings'])
            columns = [e for e in results_df.columns.tolist() if 'value' in e]
            results_df = results_df[results_df.columns.intersection(columns)]       
            for column in results_df.drop(columns='autor.value'):
                results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
            results_df = results_df.drop_duplicates().reset_index(drop=True)   
            result = results_df.to_dict('records')
            new_df.at[i, 'wikidata'] = result
            time.sleep(1)
        except (AttributeError, KeyError, ValueError):
            nodegoat_people_df.at[i, 'wikidata'] = np.nan
            time.sleep(1)
        except (HTTPError, RemoteDisconnected) as error:
            print(error)# time.sleep(61)
            time.sleep(5)
            continue
        break
new_df.to_excel(f"samizdat_osoby_{year}-{month}-{day}.xlsx", index=False)

# KROK NR 4 - Stworzenie grup dla poprawnych wyników i Oddzielenie reszty rekordów do dalszego przetwarzania

# Podzielenie na zbiory

df = nodegoat_people_df.copy().sort_values('Project_ID')
# Czeskie rekordy bez tytułów - na później
df1 = df[(df['odpytanie viafu po tytułach'].isna()) &
         (df['name_form_id'].str.contains('cz', na=False)) &
         (~df['name_form_id'].str.contains('bn', na=False))].reset_index(drop=True)

#to nie działa!!!!!!!!!!!!!!!!!!!!!!!!!!
# new_df2 = harvest_viaf_by_titles(df1, 'Project_ID', 'cz_books', '\$\$', tytuly_cz_books_df, 245, '$$a')

create_worksheet(new_sheet, 'czeskie_bez_tytulow', df1)
new_sheet.del_worksheet(new_sheet.worksheet('Arkusz1'))

df = df[~df['Project_ID'].isin(df1['Project_ID'])]
# Rekordy poprawne
# pojedynczy rekord z ID + stopień prawdopodobieństwa >= 0,75

prawdopodobienstwo = 0.75    
df2 = df.copy()
df2 = df2.groupby('Project_ID').filter(lambda x: len(x) == 1)
df2 = df2[(df2['prawdopodobieństwo'] >= prawdopodobienstwo) & (df2['prawdopodobieństwo'].notnull())]

create_worksheet(new_sheet, 'pojedyncze_ok', df2)

df = df[~df['Project_ID'].isin(df2['Project_ID'])]
# grupa rekordów z takim samym ID

prawdopodobienstwo_grupa = 0.54

df3 = df.copy()
df3['viaf'] = df3['sugestia po tytułach'].apply(lambda x: x.split('|')[1] if pd.notnull(x) else x)
a = df3.groupby('Project_ID')['viaf'].nunique().eq(1).reset_index().rename(columns={'viaf':'ok'})
df3 = df3.merge(a, on='Project_ID', how='left')
df3 = df3.groupby(['Project_ID', 'viaf', 'ok']).filter(lambda x: len(x) > 1 and any(y >= prawdopodobienstwo_grupa for y in x['prawdopodobieństwo']))

create_worksheet(new_sheet, 'grupy_ok', df3)
    
df = df[~df['Project_ID'].isin(df3['Project_ID'])]   
# pojedynczy rekord z ID + stopień prawdopodobieństwa < 0,75
df4 = df.copy()
df4 = df4.groupby('Project_ID').filter(lambda x: len(x) == 1)
df4 = df4[df4['prawdopodobieństwo'] < prawdopodobienstwo]

create_worksheet(new_sheet, f'pojedyncze_<={prawdopodobienstwo}', df4)

df = df[~df['Project_ID'].isin(df4['Project_ID'])]  
# Rekordy BN bez VIAF
df5 = df.copy()
df5 = df5[df5['odpytanie viafu po tytułach'].isna()]

create_worksheet(new_sheet, 'rekordy_BN_bez_viaf', df5)

df = df[~df['Project_ID'].isin(df5['Project_ID'])]  

# grupa rekordów które mają przypisane różne ID
create_worksheet(new_sheet, 'grupy_rozne_id_lub_niskie_prawd', df)

for worksheet in new_sheet.worksheets():
    new_sheet.batch_update({
        "requests": [
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet._properties['sheetId'],
                        "dimension": "ROWS",
                        "startIndex": 0,
                        #"endIndex": 100
                    },
                    "properties": {
                        "pixelSize": 20
                    },
                    "fields": "pixelSize"
                }
            }
        ]
    })
    
    worksheet.freeze(rows=1)
    worksheet.set_basic_filter()

# KROK 5
sheet = gc.open_by_key('175JqYyEuy9oSdk6JQqo6NgtILitXZo6Umf77YLXWLSU')
test_dict = {}
for worksheet in tqdm(sheet.worksheets()):
    title = worksheet.title
    test_df = get_as_dataframe(sheet.worksheet(title), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1)
    test_dict[title] = test_df
del test_dict['brakujące z viaf']
df_wrong = pd.concat([v for k,v in test_dict.items() if k not in ['pojedyncze_ok', 'grupy_ok']])

nodegoat_people_dict = {}
for index, name in tqdm(df_wrong.iterrows(), total=df_wrong.shape[0]):
    # index = 0
    # name = df_wrong.iloc[index,:]
    if len(name['Index_Name']) > 6:
        #index, name = nodegoat_people_tuples[7]
        index = int(index)
        url = re.sub('\s+', '%20', f"http://viaf.org/viaf/search?query=local.personalNames%20all%20%22{name['Index_Name']}%22&sortKeys=holdingscount&httpAccept=application/json")
        response = requests.get(url)
        response.encoding = 'UTF-8'
        try:
            samizdat_json = response.json()
            list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
            samizdat_json = samizdat_json['searchRetrieveResponse']['records']
            samizdat_json = [[catch(e), e['record']['recordData']['viafID']] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
            for number in list_of_numbers_of_records:
                url = re.sub('\s+', '%20', f"http://viaf.org/viaf/search?query=local.personalNames%20all%20%22{name['Index_Name']}%22&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                response = requests.get(url)
                response.encoding = 'UTF-8'
                samizdat_json_next = response.json()
                samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                samizdat_json_next = [[catch(e), e['record']['recordData']['viafID']] for e in samizdat_json_next if e['record']['recordData']['nameType'] == 'Personal']
                samizdat_json += samizdat_json_next
            #tutaj nadpisują się project ID, dlatego nie ma kilku nazewnictw dla osoby - przywrócić wcześniejsze ify
            nodegoat_people_dict[index] = {'Project_ID':name['Project_ID'], 'samizdat_name':name['Index_Name']}
            nodegoat_people_dict[index]['viaf'] = samizdat_json
        except KeyError:
            nodegoat_people_dict[index] = {'Project_ID':name['Project_ID'], 'samizdat_name':name['Index_Name']}
            nodegoat_people_dict[index]['viaf'] = []

for person_k, person_v in tqdm(nodegoat_people_dict.items()):
    # person_k = 0
    # person_v = nodegoat_people_dict[person_k]
    persons = [e[0].split('❦') for e in person_v['viaf']]
    for index, names in enumerate(persons):
        # index = 0
        # names = persons[index]
        list_of_similarity_indexes = []
        for name in names:
            list_of_similarity_indexes.append(difflib.SequenceMatcher(a=person_v['samizdat_name'], b=name).ratio())    
        nodegoat_people_dict[person_k]['viaf'][index].append(max(list_of_similarity_indexes))
            
# with open(f"samizdat_people_{year}-{month}-{day}.json", 'w', encoding='utf-8') as f: 
#     json.dump(nodegoat_people_dict, f, ensure_ascii=False, indent=4)    
    
# with open('samizdat_people_2021-06-09.json', 'r', encoding='utf-8') as json_file:
#     nodegoat_people_dict = json.load(json_file)    
    
nodegoat_people_df = [value for value in nodegoat_people_dict.values()]    
nodegoat_people_df = pd.json_normalize(nodegoat_people_df)
nodegoat_people_df = nodegoat_people_df.explode('viaf').reset_index(drop=True)
nodegoat_people_df['viaf id'] = nodegoat_people_df['viaf'].apply(lambda x: x[1] if type(x) != float else np.nan)
nodegoat_people_df['similarity'] = nodegoat_people_df['viaf'].apply(lambda x: x[-1] if type(x) != float else np.nan)
nodegoat_people_df = nodegoat_people_df.sort_values(['Project_ID', 'similarity'], ascending=[True, False])
# nodegoat_people_df = nodegoat_people_df[(nodegoat_people_df['similarity'] >= prawdopodobienstwo) | (nodegoat_people_df['similarity'].isna())]

# new_sheet = gc.open_by_key('175JqYyEuy9oSdk6JQqo6NgtILitXZo6Umf77YLXWLSU')
# create_worksheet(new_sheet, 'brakujące z viaf', nodegoat_people_df)


# #TUTAJ 21.06.2021
# nodegoat_people_sheet = gc.open_by_key('175JqYyEuy9oSdk6JQqo6NgtILitXZo6Umf77YLXWLSU')
# nodegoat_people_sheet.worksheets()
# samizdat_dict = {}
# for ws in nodegoat_people_sheet.worksheets():
#     samizdat_dict[ws.title] = get_as_dataframe(ws, evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()

nodegoat_people_df['wikidata'] = np.nan
nodegoat_people_df['wikidata'] = nodegoat_people_df['wikidata'].astype(object)
url = 'https://query.wikidata.org/sparql'
for i, row in tqdm(nodegoat_people_df.iterrows(), total=nodegoat_people_df.shape[0]):
    while True:
        try:
            viaf = row['viaf id']
            sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
              ?autor wdt:P214 "{viaf}" ;
              optional {{ ?autor wdt:P19 ?birthplace . }}
              optional {{ ?autor wdt:P569 ?birthdate . }}
              optional {{ ?autor wdt:P570 ?deathdate . }}
              optional {{ ?autor wdt:P20 ?deathplace . }}
              optional {{ ?autor wdt:P21 ?sex . }}
              optional {{ ?autor wdt:P106 ?occupation . }}
              optional {{ ?autor wdt:P742 ?pseudonym . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
            results = requests.get(url, params = {'format': 'json', 'query': sparql_query})
            results = results.json()
            results_df = pd.json_normalize(results['results']['bindings'])
            columns = [e for e in results_df.columns.tolist() if 'value' in e]
            results_df = results_df[results_df.columns.intersection(columns)]       
            for column in results_df.drop(columns='autor.value'):
                results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
            results_df = results_df.drop_duplicates().reset_index(drop=True)   
            result = results_df.to_dict('records')
            nodegoat_people_df.at[i, 'wikidata'] = result
            time.sleep(1)
        except (AttributeError, KeyError, ValueError):
            nodegoat_people_df.at[i, 'wikidata'] = np.nan
            time.sleep(1)
        except (HTTPError, RemoteDisconnected) as error:
            print(error)# time.sleep(61)
            time.sleep(5)
            continue
        break
nodegoat_people_df.to_excel(f"samizdat_osoby_{year}-{month}-{day}.xlsx", index=False)

samizdat_people_dict = {}
samizdat_people_dict['pojedyncze ok'] = test_dict['pojedyncze_ok']
samizdat_people_dict['grupy_ok'] = test_dict['grupy_ok']

jedna_osoba = nodegoat_people_df.groupby('Project_ID').filter(lambda x: len(x) == 1)

samizdat_people_dict['osoby_z_jednym_wierszem'] = jedna_osoba

nodegoat_people_df = nodegoat_people_df[~nodegoat_people_df['Project_ID'].isin(jedna_osoba['Project_ID'])]

grouped = nodegoat_people_df.groupby('Project_ID')

nodegoat_people_df = pd.DataFrame()
for name, group in tqdm(grouped, total=len(grouped)):
    # group = grouped.get_group(33)
    scores_similarity = {e:i+1 for i, e in enumerate(sorted(group['similarity'].unique()))}
    scores_similarity = pd.DataFrame.from_dict(scores_similarity, orient='index').reset_index().rename(columns={'index':'similarity', 0:'similarity_score'})
    group = pd.merge(group, scores_similarity, how='left', on='similarity')
    group['flags'] = group['viaf'].apply(lambda x: sum([int(e.split('|')[-1]) for e in x[0].split('❦')]) if isinstance(x, list) else 0)
    scores_flags = {e:i+1 for i, e in enumerate(sorted(group['flags'].unique()))}
    scores_flags = pd.DataFrame.from_dict(scores_flags, orient='index').reset_index().rename(columns={'index':'flags', 0:'flags_score'})
    group = pd.merge(group, scores_flags, how='left', on='flags')
    group['score'] = group['flags_score'] + group['similarity_score']
    group.sort_values(['score', 'similarity', 'flags'], ascending=[False, False, False], inplace=True)
    nodegoat_people_df = nodegoat_people_df.append(group)

samizdat_people_dict['reszta'] = nodegoat_people_df

new_sheet = gc.create(f'samizdat_osoby_{year}-{month}-{day}', '1UdglvjjX4r2Hzh5BIAr8FjPuWV89NVs6')
for k,v in tqdm(samizdat_people_dict.items()):
    try:
        set_with_dataframe(new_sheet.worksheet(k), v)
    except gs.WorksheetNotFound:
        new_sheet.add_worksheet(title=k, rows="100", cols="20")
        set_with_dataframe(new_sheet.worksheet(k), v)
    
new_sheet.del_worksheet(new_sheet.worksheet('Arkusz1'))

for worksheet in new_sheet.worksheets():
    new_sheet.batch_update({
        "requests": [
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet._properties['sheetId'],
                        "dimension": "ROWS",
                        "startIndex": 0,
                        #"endIndex": 100
                    },
                    "properties": {
                        "pixelSize": 20
                    },
                    "fields": "pixelSize"
                }
            }
        ]
    })   
    worksheet.freeze(rows=1)
    worksheet.set_basic_filter()


sheet = gc.open_by_key('1TFNZboZaSHZv0FJXCGgcLs7yN47hbFirXFgIa74KupM')
test_dict = {}
for worksheet in tqdm(sheet.worksheets()):
    title = worksheet.title
    test_df = get_as_dataframe(sheet.worksheet(title), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1)
    test_dict[title] = test_df

order = ['Project_ID', 'Index_Name', 'name_form_id', 'odpytanie viafu po tytułach', 'sugestia po tytułach', 'viaf', 'name', 'pseudym of', 'score', 'similarity', 'flags', 'viaf_ID', 'wikidata_ID', 'autor.value', 'birthdate.value', 'deathdate.value', 'birthplaceLabel.value', 'deathplaceLabel.value', 'sexLabel.value', 'pseudonym.value', 'occupationLabel.value']

for k,v in tqdm(test_dict.items()):
    try:
        v['wikidata'] = v['wikidata'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)
    except ValueError:
        pass
    if k in ['grupy_ok']:
        v = pd.concat([v.drop(['wikidata'], axis=1), v['wikidata'].apply(pd.Series)], axis=1).rename(columns={'prawdopodobieństwo':'similarity', 'viaf':'viaf_ID'})
    elif k == 'osoby_z_jednym_wierszem':
        v = pd.concat([v.drop(['wikidata'], axis=1), v['wikidata'].apply(pd.Series)], axis=1).rename(columns={'samizdat_name':'Index_Name', 'viaf id':'viaf_ID'})
    elif k == 'pojedyncze ok':
        v = pd.concat([v.drop(['wikidata'], axis=1), v['wikidata'].apply(pd.Series)], axis=1).rename(columns={'prawdopodobieństwo':'similarity'})
        v['viaf_ID'] = v['sugestia po tytułach'].apply(lambda x: x.split('|')[1])
    elif k == 'reszta':
        v = pd.concat([v.drop(['wikidata'], axis=1), v['wikidata'].apply(pd.Series)], axis=1).rename(columns={'samizdat_name':'Index_Name', 'viaf id':'viaf_ID'})
    v = pd.concat([v.drop([0], axis=1), v[0].apply(pd.Series)], axis=1).drop(columns=0)
    v = v.reindex(columns=order)
    v['wikidata_ID'] = v['autor.value'].apply(lambda x: re.findall('Q\d+$', x)[0] if pd.notnull(x) else x)
    v.dropna(how='all', axis=1, inplace=True)
    v['decyzja'] = np.nan
    test_dict[k] = v

new_sheet = gc.create(f'samizdat_osoby_{year}-{month}-{day}', '1UdglvjjX4r2Hzh5BIAr8FjPuWV89NVs6')
for k,v in tqdm(test_dict.items()):
    try:
        set_with_dataframe(new_sheet.worksheet(k), v)
    except gs.WorksheetNotFound:
        new_sheet.add_worksheet(title=k, rows="100", cols="20")
        set_with_dataframe(new_sheet.worksheet(k), v)
    
new_sheet.del_worksheet(new_sheet.worksheet('Arkusz1'))

for worksheet in new_sheet.worksheets():
    new_sheet.batch_update({
        "requests": [
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet._properties['sheetId'],
                        "dimension": "ROWS",
                        "startIndex": 0,
                        #"endIndex": 100
                    },
                    "properties": {
                        "pixelSize": 20
                    },
                    "fields": "pixelSize"
                }
            }
        ]
    })   
    worksheet.freeze(rows=1)
    worksheet.set_basic_filter()































# for k,v in samizdat_dict.items():
for i, row in tqdm(nodegoat_people_df.iterrows(), total=nodegoat_people_df.shape[0]):
    while True:
        try:
            name = ' '.join(row['Index_Name'].split(', ')[::-1])
            sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
              ?autor wdt:P31 wd:Q5.
              ?autor ?label "{name}" .
              optional {{ ?autor wdt:P19 ?birthplace . }}
              optional {{ ?autor wdt:P569 ?birthdate . }}
              optional {{ ?autor wdt:P570 ?deathdate . }}
              optional {{ ?autor wdt:P20 ?deathplace . }}
              optional {{ ?autor wdt:P21 ?sex . }}
              optional {{ ?autor wdt:P106 ?occupation . }}
              optional {{ ?autor wdt:P742 ?pseudonym . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
            results = requests.get(url, params = {'format': 'json', 'query': sparql_query})
            results = results.json()
            results_df = pd.json_normalize(results['results']['bindings'])
            columns = [e for e in results_df.columns.tolist() if 'value' in e]
            results_df = results_df[results_df.columns.intersection(columns)]       
            for column in results_df.drop(columns='autor.value'):
                results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
            results_df = results_df.drop_duplicates().reset_index(drop=True)   
            result = results_df.to_dict('records')
            nodegoat_people_df.at[i, 'wikidata'] = result
            time.sleep(2)
        except (AttributeError, KeyError, ValueError):
            nodegoat_people_df.at[i, 'wikidata'] = np.nan
            time.sleep(2)
        except (HTTPError, RemoteDisconnected) as error:
            print(error)# time.sleep(61)
            time.sleep(5)
            continue
        break
    
    
    
    try:
        samizdat_dict[k]['viafID'] = samizdat_dict[k]['sugestia po tytułach'].apply(lambda x: x.split('|')[1])
    except KeyError:
        pass
    if k in ['pojedyncze_ok', 'grupy_ok']:
        samizdat_dict[k]['viafREV'] = np.nan
    else:
        # k = 'czeskie_bez_tytulow'
        samizdat_dict[k]['wikidata'] = np.nan
        samizdat_dict[k]['wikidata'] = samizdat_dict[k]['wikidata'].astype(object)
        for i, row in tqdm(samizdat_dict[k].iterrows(), total=samizdat_dict[k].shape[0]):
            # i = 7
            # row = samizdat_dict[k].iloc[7,:]
            
            while True:
                try:
                    name = ' '.join(row['Index_Name'].split(', ')[::-1])
                    sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
                      ?autor wdt:P31 wd:Q5.
                      ?autor ?label "{name}" .
                      optional {{ ?autor wdt:P19 ?birthplace . }}
                      optional {{ ?autor wdt:P569 ?birthdate . }}
                      optional {{ ?autor wdt:P570 ?deathdate . }}
                      optional {{ ?autor wdt:P20 ?deathplace . }}
                      optional {{ ?autor wdt:P21 ?sex . }}
                      optional {{ ?autor wdt:P106 ?occupation . }}
                      optional {{ ?autor wdt:P742 ?pseudonym . }}
                    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
                    results = requests.get(url, params = {'format': 'json', 'query': sparql_query})
                    results = results.json()
                    results_df = pd.json_normalize(results['results']['bindings'])
                    columns = [e for e in results_df.columns.tolist() if 'value' in e]
                    results_df = results_df[results_df.columns.intersection(columns)]       
                    for column in results_df.drop(columns='autor.value'):
                        results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
                    results_df = results_df.drop_duplicates().reset_index(drop=True)   
                    result = results_df.to_dict('records')
                    samizdat_dict[k].at[i, 'wikidata'] = result
                    time.sleep(2)
                except (AttributeError, KeyError, ValueError):
                    samizdat_dict[k].at[i, 'wikidata'] = np.nan
                    time.sleep(2)
                except (HTTPError, RemoteDisconnected) as error:
                    print(error)# time.sleep(61)
                    time.sleep(5)
                    continue
                break
    try:
        samizdat_dict[k]['wikidataID'] = samizdat_dict[k]['wikidata'].apply(lambda x: get_wikidataID(x))
    except KeyError:
        pass        
    
    
###    
for key, df in samizdat_dict.items():
    try:
        df.to_csv(key+'.csv', index=False)    
    except OSError:
        key = ''.join([e for e in key if e.isalnum()])
        df.to_csv(key+'.csv', index=False)    
        
files = ["C:/Users/Cezary/Documents/IBL-PAN-Python/czeskie_bez_tytulow.csv",
"C:/Users/Cezary/Documents/IBL-PAN-Python/grupy_ok.csv",
"C:/Users/Cezary/Documents/IBL-PAN-Python/grupy_rozne_id_lub_niskie_prawd.csv",
"C:/Users/Cezary/Documents/IBL-PAN-Python/pojedyncze_ok.csv",
"C:/Users/Cezary/Documents/IBL-PAN-Python/pojedyncze075.csv",
"C:/Users/Cezary/Documents/IBL-PAN-Python/rekordy_BN_bez_viaf.csv"]   
samizdat_dict = {}
for file in files:
    name = re.findall('(?<=Python\/)(.+?)(?=\.csv)', file)[0]
    samizdat_dict[name] = pd.read_csv(file)
        
#podzielić dane na grupy!!!    
new_samizdat_dict = {}    
for k,v in samizdat_dict.items():
    if k in ['grupy_ok', 'pojedyncze_ok']:
        new_samizdat_dict[k] = v
    else:
        if 'reszta' not in new_samizdat_dict:
            new_samizdat_dict['reszta'] = v
        else:
            new_samizdat_dict['reszta'] = pd.concat([new_samizdat_dict['reszta'], v])
        
grouped_df = new_samizdat_dict['reszta'].groupby('Project_ID')
for name, group in tqdm(grouped_df, total=len(grouped_df)):
    # name = 476
    # group = grouped_df.get_group(name)
    if group.shape[0] == 1:
        if all(pd.notnull(group['viafID'])):
            if 'osoby z jednym wierszem i identyfikatorem viaf' not in new_samizdat_dict:
                new_samizdat_dict['osoby z jednym wierszem i identyfikatorem viaf'] = group
            else:
                new_samizdat_dict['osoby z jednym wierszem i identyfikatorem viaf'] = pd.concat([new_samizdat_dict['osoby z jednym wierszem i identyfikatorem viaf'], group])
        else:
            if 'osoby z pustym viafem' not in new_samizdat_dict:
                new_samizdat_dict['osoby z pustym viafem'] = group
            else:
                new_samizdat_dict['osoby z pustym viafem'] = pd.concat([new_samizdat_dict['osoby z pustym viafem'], group])      
    elif group.shape[0] > 1 and group['wikidata'].notna().sum() == 1:
        if 'grupa z jedną wikidatą' not in new_samizdat_dict:
            new_samizdat_dict['grupa z jedną wikidatą'] = group
        else:
            new_samizdat_dict['grupa z jedną wikidatą'] = pd.concat([new_samizdat_dict['grupa z jedną wikidatą'], group])
    elif group.shape[0] > 1 and group['wikidata'].notna().sum() > 1:
        if 'grupa z wieloma wikidatami' not in new_samizdat_dict:
            new_samizdat_dict['grupa z wieloma wikidatami'] = group
        else:
            new_samizdat_dict['grupa z wieloma wikidatami'] = pd.concat([new_samizdat_dict['grupa z wieloma wikidatami'], group])
    elif group.shape[0] > 1 and group['wikidata'].notna().sum() == 0:
        if 'grupa bez wikidaty' not in new_samizdat_dict:
            new_samizdat_dict['grupa bez wikidaty'] = group
        else:
            new_samizdat_dict['grupa bez wikidaty'] = pd.concat([new_samizdat_dict['grupa bez wikidaty'], group])
    else:
        if 'pozostałe' not in new_samizdat_dict:
            new_samizdat_dict['pozostałe'] = group
        else:
            new_samizdat_dict['pozostałe'] = pd.concat([new_samizdat_dict['pozostałe'], group])
del new_samizdat_dict['reszta']    


new_sheet = gc.create(f'samizdat_osoby_{year}-{month}-{day}', '1UdglvjjX4r2Hzh5BIAr8FjPuWV89NVs6')
for k,v in tqdm(new_samizdat_dict.items(), total=len(new_samizdat_dict)):
    try:
        v = v.drop(columns='viaf')
    except KeyError:
        pass
    create_worksheet(new_sheet, k, v)
    
new_sheet.del_worksheet(new_sheet.worksheet('Arkusz1'))        
for worksheet in new_sheet.worksheets():
    new_sheet.batch_update({
        "requests": [
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet._properties['sheetId'],
                        "dimension": "ROWS",
                        "startIndex": 0,
                        #"endIndex": 100
                    },
                    "properties": {
                        "pixelSize": 20
                    },
                    "fields": "pixelSize"
                }
            }
        ]
    })
    
    worksheet.freeze(rows=1)
    worksheet.set_basic_filter()    


    





type(samizdat_dict['grupy_ok'].at[0, 'wikidata'])

type(samizdat_dict['rekordy_BN_bez_viaf'].at[9, 'wikidata'])

oo = samizdat_dict['rekordy_BN_bez_viaf'].at[9, 'wikidata']
isinstance(oo, str)








































def catch(e):
       try:
           return e['record']['recordData']['mainHeadings']['data'][0]['text']
       except KeyError:
           return e['record']['recordData']['mainHeadings']['data']['text']

#dodać do tego kroku podobieństwo difflibu czy później?
       
nodegoat_people_dict = {}
for index, name in tqdm(nodegoat_people_tuples):
    #index, name = nodegoat_people_tuples[7]
    index = int(index)
    url = re.sub('\s+', '%20', f"http://viaf.org/viaf/search?query=local.personalNames%20all%20%22{name}%22&sortKeys=holdingscount&httpAccept=application/json")
    response = requests.get(url)
    response.encoding = 'UTF-8'
    try:
        samizdat_json = response.json()
        list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
        samizdat_json = samizdat_json['searchRetrieveResponse']['records']
        samizdat_json = [[catch(e), e['record']['recordData']['viafID']] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
        for number in list_of_numbers_of_records:
            url = re.sub('\s+', '%20', f"http://viaf.org/viaf/search?query=local.personalNames%20all%20%22{name}%22&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
            response = requests.get(url)
            response.encoding = 'UTF-8'
            samizdat_json_next = response.json()
            samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
            samizdat_json_next = [[catch(e), e['record']['recordData']['viafID']] for e in samizdat_json_next if e['record']['recordData']['nameType'] == 'Personal']
            samizdat_json += samizdat_json_next
        
        if index in nodegoat_people_dict:
            nodegoat_people_dict[index]['samizdat_name'].append(name)
            nodegoat_people_dict[index]['viaf'] += samizdat_json
        else:
            nodegoat_people_dict[index] = {'samizdatID':index, 'samizdat_name':[name]}
            nodegoat_people_dict[index]['viaf'] = samizdat_json
    except KeyError:
        if index in nodegoat_people_dict:
            nodegoat_people_dict[index]['samizdat_name'].append(name)
        else:
            nodegoat_people_dict[index] = {'samizdatID':index, 'samizdat_name':[name]}
            nodegoat_people_dict[index]['viaf'] = []

nodegoat_people_df = [value for value in nodegoat_people_dict.values()]    
nodegoat_people_df = pd.json_normalize(nodegoat_people_df)
nodegoat_people_df = nodegoat_people_df.explode('viaf').reset_index(drop=True)
nodegoat_people_df['viaf id'] = nodegoat_people_df['viaf'].apply(lambda x: x[1] if type(x) != float else np.nan)
nodegoat_people_df['wikidata'] = np.nan
nodegoat_people_df['wikidata'] = nodegoat_people_df['wikidata'].astype(object)

#del nodegoat_people_df['wikidata']    
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")    
        
for i, row in tqdm(nodegoat_people_df.iterrows(), total=nodegoat_people_df.shape[0]):
    try:
        viaf = row['viaf id']
        sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
          ?autor wdt:P214 "{viaf}" ;
          optional {{ ?autor wdt:P19 ?birthplace . }}
          optional {{ ?autor wdt:P569 ?birthdate . }}
          optional {{ ?autor wdt:P570 ?deathdate . }}
          optional {{ ?autor wdt:P20 ?deathplace . }}
          optional {{ ?autor wdt:P21 ?sex . }}
          optional {{ ?autor wdt:P106 ?occupation . }}
          optional {{ ?autor wdt:P742 ?pseudonym . }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.json_normalize(results['results']['bindings'])
        columns = [e for e in results_df.columns.tolist() if 'value' in e]
        results_df = results_df[results_df.columns.intersection(columns)]       
        for column in results_df.drop(columns='autor.value'):
            results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
        results_df = results_df.drop_duplicates().reset_index(drop=True)   
        result = results_df.to_dict('records')
        nodegoat_people_df.at[i, 'wikidata'] = result
    except (KeyError, TypeError):
        nodegoat_people_df.at[i, 'wikidata'] = np.nan
    except (HTTPError, RemoteDisconnected) as error:
        print(error)
        # time.sleep(61)
        time.sleep(5)
        pass
    
nodegoat_viaf_wikidata_df = nodegoat_people_df.groupby('samizdatID').filter(lambda x: len(x) == 1)
nodegoat_viaf_wikidata_df = nodegoat_viaf_wikidata_df[nodegoat_viaf_wikidata_df['wikidata'].notnull()].drop(columns=['viaf id'])     

nodegoat_people_df = nodegoat_people_df[~nodegoat_people_df['samizdatID'].isin(nodegoat_viaf_wikidata_df['samizdatID'])].drop(columns=['viaf id'])
nodegoat_people_df = nodegoat_people_df[['samizdatID', 'samizdat_name', 'viaf', 'wikidata']]




df = [e for e in df if max(e[-1])]

df = pd.DataFrame(df, columns=['name', 'viaf', 'similarity'])

df = pd.concat([pd.DataFrame.from_dict(slownik[e], orient='index') for e in slownik])  


return '❦'.join([a['text'] for a in e['record']['recordData']['mainHeadings']['data']])        
                















# test = gc.open_by_key('1VNzdJzCaQzPYKLySUD3Jc5D8U-B4c_T2SE68ybc9c9o')
# test = get_as_dataframe(test.worksheet('po tytułach'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()

test = test.groupby('samizdatID')
new_df = pd.DataFrame()
# for i, row in tqdm(test.iterrows(), total=test.shape[0]):
for name, group in tqdm(test, total=len(test)):
    # group = test.get_group(477)
    group = group.reset_index(drop=True)
    for i, row in group.iterrows():
        # i = 0
        # row = group.iloc[i,:]
        locations = row['name_form_id'].split('|')
        viaf = row['viaf'][1]
        # viaf = '101882692'
        for location in locations:
            # location = locations[0]
            location = location.split('-')
            if location[0] == 'bn_books':
                title = tytuly_bn_df[tytuly_bn_df['id'] == int(location[1])][200].reset_index(drop=True)[0]
                title = marc_parser_dict_for_field(title, '\%')['%a']
                # if '%e' in title:
                #     title = marc_parser_dict_for_field(title, '\%')['%a'] + ' ' + marc_parser_dict_for_field(title, '\%')['%e']
                # else:
                #     title = marc_parser_dict_for_field(title, '\%')['%a']
                # url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                try:
                    url = f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&httpAccept=application/json"
                    response = requests.get(url)
                    response.encoding = 'UTF-8'
                    samizdat_json = response.json()  
                    list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
                    samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                    for number in list_of_numbers_of_records:
                        url = re.sub('\s+', '%20', f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                        response = requests.get(url)
                        response.encoding = 'UTF-8'
                        samizdat_json_next = response.json()
                        samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                        samizdat_json += samizdat_json_next
                except ValueError:
                    try:
                        url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                        response = requests.get(url)
                        response.encoding = 'UTF-8'
                        samizdat_json = response.json()  
                        list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
                        samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                        for number in list_of_numbers_of_records:
                            url = re.sub('\s+', '%20', f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                            response = requests.get(url)
                            response.encoding = 'UTF-8'
                            samizdat_json_next = response.json()
                            samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                            samizdat_json += samizdat_json_next
                    except ValueError:
                        # break
                        pass
                except KeyError:
                    # break  
                    pass
                    #TUTAJ albo personal albo work
                try:
                    samizdat_personal = [e['record']['recordData']['viafID'] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
                    samizdat_json_selected = [e for e in samizdat_personal if e == viaf][0]
                    group.at[i, 'odpytanie po tytułach'] = samizdat_json_selected
                    break
                except IndexError:
                    try:
                        samizdat_json_title = [e['record']['recordData']['titles']['author']['@id'].split('|')[-1] for e in samizdat_json]
                        samizdat_json_selected = [e for e in samizdat_json_title if e == viaf][0]
                        group.at[i, 'odpytanie po tytułach'] = samizdat_json_selected
                        break
                    except (IndexError, KeyError):
                        continue
    new_df = new_df.append(group)
    
    
ttt = new_df[new_df['odpytanie po tytułach'].notnull()]    
    
    
    
    
    
    
    
    
    
    try:
        viaf = row['viaf'][1]
        # viaf = '101882692'
        list_for_titles = []
        for location in locations:
            # location = locations[0]
            location = location.split('-')
            if location[0] == 'bn_books':
                title = tytuly_bn_df[tytuly_bn_df['id'] == int(location[1])][200].reset_index(drop=True)[0]
                if '%e' in title:
                    title = marc_parser_dict_for_field(title, '\%')['%a'] + ' ' + marc_parser_dict_for_field(title, '\%')['%e']
                else:
                    title = marc_parser_dict_for_field(title, '\%')['%a']
                # url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                url = f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&httpAccept=application/json"
                response = requests.get(url)
                response.encoding = 'UTF-8'
                try:
                    samizdat_json = response.json()  
                    list_of_numbers_of_records = [e for e in range(int(samizdat_json['searchRetrieveResponse']['numberOfRecords']))[11::10] if e <= 100]
                    samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                    for number in list_of_numbers_of_records:
                        url = re.sub('\s+', '%20', f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&startRecord={number}&httpAccept=application/json")
                        response = requests.get(url)
                        response.encoding = 'UTF-8'
                        samizdat_json_next = response.json()
                        samizdat_json_next = samizdat_json_next['searchRetrieveResponse']['records']
                        samizdat_json += samizdat_json_next
                    samizdat_json = [e['record']['recordData']['viafID'] for e in samizdat_json]
                except IndexError:
                    pass
                try:
                    samizdat_json_selected = [e for e in samizdat_json if e == viaf][0]
                    test.at[i, 'odpytanie po tytułach'] = samizdat_json_selected
                    break
                except (KeyError, ValueError):
                    pass
    except:
        print('nie ma viaf')
        pass

                    
                    
                    
                    
                    
                    try:
                        samizdat_json_selected = [e for e in samizdat_json if e[1] == viaf][0]
                        if type(samizdat_json_selected[0]) == list:
                            nazwa = [e['text'] for e in samizdat_json_selected[0]]
                        elif type(samizdat_json_selected[0]) == dict:
                            nazwa = samizdat_json_selected[0]['text']
                        
                        if type(samizdat_json_selected[2]) == list:
                            tytuly = [e['title'] for e in samizdat_json_selected[2]]
                        elif type(samizdat_json_selected[2]) == dict:
                            tytuly = samizdat_json_selected[2]['title']
                            
                        samizdat_dict = str({'viaf':viaf,'nazwa':nazwa, 'tytuły':tytuly})
                        list_for_titles.append(samizdat_dict)
                        #test.at[i, 'odpytanie po tytułach'] = samizdat_dict
                    except IndexError:
                        test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
                        pass
                except (KeyError, ValueError):
                    pass
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
            list_for_titles = list(set(list_for_titles))
            if list_for_titles:
                test.at[i, 'odpytanie po tytułach'] = list_for_titles
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do usunięcia'
    except TypeError:
        list_for_titles = []
        for location in locations:
            #location = locations[0]
            location = location.split('-')
            if location[0] == 'bn_books':
                title = tytuly_bn_df[tytuly_bn_df['id'] == int(location[1])][200].reset_index(drop=True)[0]
                if '%e' in title:
                    title = marc_parser_dict_for_field(title, '\%')['%a'] + ' ' + marc_parser_dict_for_field(title, '\%')['%e']
                else:
                    title = marc_parser_dict_for_field(title, '\%')['%a']
                url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                response = requests.get(url)
                response.encoding = 'UTF-8'
                try:
                    samizdat_json = response.json()  
                    samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                    samizdat_json = [[e['record']['recordData']['mainHeadings']['data'], e['record']['recordData']['viafID'], e['record']['recordData']['titles']['work'], len(e['record']['recordData']['sources']['source'])] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
                    if len(samizdat_json) == 1:
                        samizdat_json_selected = samizdat_json[0]
                        if type(samizdat_json_selected[0]) == list:
                            nazwa = [e['text'] for e in samizdat_json_selected[0]]
                        elif type(samizdat_json_selected[0]) == dict:
                            nazwa = samizdat_json_selected[0]['text']
                        
                        if type(samizdat_json_selected[2]) == list:
                            tytuly = [e['title'] for e in samizdat_json_selected[2]]
                        elif type(samizdat_json_selected[2]) == dict:
                            tytuly = samizdat_json_selected[2]['title']
                            
                        samizdat_dict = str({'viaf':viaf,'nazwa':nazwa, 'tytuły':tytuly})
                        list_for_titles.append(samizdat_dict)
                    else:
                        test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
                        pass
                except (KeyError, ValueError, TypeError):
                    pass
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
            list_for_titles = list(set(list_for_titles))
            if list_for_titles:
                test.at[i, 'odpytanie po tytułach'] = list_for_titles
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do usunięcia'


























    
    
    samizdat_json = [[e[0][0]['text'], f] for e, f in samizdat_json]
    type(samizdat_json[0][0][0])
    samizdat_json[0][0][0]['text']
    
    samizdat_json = [[e['record']['recordData']['mainHeadings']['data'], e['record']['recordData']['viafID'], e['record']['recordData']['titles']['work'], len(e['record']['recordData']['sources']['source'])] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']





samizdat_viaf = pd.DataFrame()
for index, row in samizdat_people.iterrows():

    search_name = row['Index_Name']
    
    print(str(index+1) + '/' + str(len(samizdat_people)))
    connection_no = 1
    while True:
        try:
            people_links = []
            while len(people_links) == 0 and len(search_name) > 0:
                url = re.sub('\s+', '%20', f"http://viaf.org/viaf/search?query=local.personalNames%20all%20%22{search_name}%22&sortKeys=holdingscount&recordSchema=BriefVIAF")
                response = requests.get(url)
                response.encoding = 'UTF-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                people_links = soup.findAll('a', attrs={'href': re.compile("viaf/\d+")})
                if len(people_links) == 0:
                    search_name = ' '.join(search_name.split(' ')[:-1])
                
            if len(people_links) > 0:
                viaf_people = []
                for people in people_links:
                    person_name = re.split('â\x80\x8e|\u200e ', re.sub('\s+', ' ', people.text).strip())
                    person_link = re.sub(r'(.+?)(\#.+$)', r'http://viaf.org\1viaf.xml', people['href'].strip())
                    person_link = [person_link] * len(person_name)
# =============================================================================
#                     libraries = str(people).split('<br/>')
#                     libraries = [re.sub('(.+)(\<span.*+$)', r'\2', s.replace('\n', ' ')) for s in libraries if 'span' in s]
#                     single_record = list(zip(person_name, person_link, libraries))
#                     viaf_people += single_record
#                 viaf_people = pd.DataFrame(viaf_people, columns=['viaf name', 'viaf', 'libraries'])
# =============================================================================
                    single_record = list(zip(person_name, person_link))
                    viaf_people += single_record
                viaf_people = pd.DataFrame(viaf_people, columns=['viaf name', 'viaf'])
                viaf_people['Project_ID'] = row['Project_ID']
                viaf_people['Index_Name'] = row['Index_Name']
                viaf_people['search name'] = search_name
                for ind, vname in viaf_people.iterrows():
                    viaf_people.at[ind, 'cosine'] = get_cosine_result(vname['viaf name'], vname['search name'])
        
                if viaf_people['cosine'].max() >= 0.5:
                    viaf_people = viaf_people[viaf_people['cosine'] >= 0.5]
                else:
                    viaf_people = viaf_people[viaf_people['cosine'] == viaf_people['cosine'].max()]
                    
                viaf_people = viaf_people.drop(columns='search name').drop_duplicates().reset_index(drop=True) 
            
                samizdat_viaf = samizdat_viaf.append(viaf_people)
            else:
                viaf_people = pd.DataFrame({'viaf name': ['brak'], 'viaf': ['brak'], 'Project_ID': [row['Project_ID']], 'Index_Name': [row['Index_Name']]})
                samizdat_viaf = samizdat_viaf.append(viaf_people)
        except (IndexError, KeyError):
            pass
        except requests.exceptions.ConnectionError:
            print(connection_no)
            connection_no += 1
            time.sleep(300)
            continue
        break

samizdat_viaf = samizdat_viaf[samizdat_viaf['viaf name'] != 'brak']
for column in samizdat_viaf.drop(columns=['viaf', 'Project_ID']):
    samizdat_viaf[column] = samizdat_viaf.groupby(['viaf', 'Project_ID'])[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
samizdat_viaf = samizdat_viaf.drop_duplicates().reset_index(drop=True) 


nodegoat_people_dict = {}
for index, name in tqdm(nodegoat_people_tuples):
    index = int(index)
    url = f"http://www.viaf.org/viaf/AutoSuggest?query={name}"
    response = requests.get(url)
    response.encoding = 'UTF-8'
    try:
        samizdat_json = response.json()
        samizdat_json = [e for e in samizdat_json['result'] if e['nametype'] == 'personal']
        if index in nodegoat_people_dict:
            nodegoat_people_dict[index]['samizdat_name'].append(name)
            nodegoat_people_dict[index]['viaf'] += [[e['displayForm'], e['viafid'], e['score']] for e in samizdat_json]
        else:
            nodegoat_people_dict[index] = {'samizdatID':index, 'samizdat_name':[name]}
            nodegoat_people_dict[index]['viaf'] = [[e['displayForm'], e['viafid'], e['score']] for e in samizdat_json]
    except (ValueError, TypeError):
        if index in nodegoat_people_dict:
            nodegoat_people_dict[index]['samizdat_name'].append(name)
        else:
            nodegoat_people_dict[index] = {'samizdatID':index, 'samizdat_name':[name]}
            nodegoat_people_dict[index]['viaf'] = []

for k, v in nodegoat_people_dict.items():
    nodegoat_people_dict[k]['viaf'] = [list(x) for x in set(tuple(x) for x in nodegoat_people_dict[k]['viaf'])]

nodegoat_people_df = [value for value in nodegoat_people_dict.values()]    
nodegoat_people_df = pd.json_normalize(nodegoat_people_df)
nodegoat_people_df = nodegoat_people_df.explode('viaf').reset_index(drop=True)
nodegoat_people_df['viaf id'] = nodegoat_people_df['viaf'].apply(lambda x: x[1] if type(x) != float else np.nan)
nodegoat_people_df['viaf score'] = nodegoat_people_df['viaf'].apply(lambda x: int(x[-1]) if type(x) != float else np.nan)
nodegoat_people_df = nodegoat_people_df.sort_values(['samizdatID', 'viaf score'], ascending=[True, False]).groupby(['samizdatID', 'viaf id'], dropna=False).head(1)

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")    
        
for i, row in tqdm(nodegoat_people_df.iterrows(), total=nodegoat_people_df.shape[0]):
    try:
        viaf = row['viaf'][1]
        sparql_query = f"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT distinct ?autor ?autorLabel ?birthplaceLabel ?deathplaceLabel ?birthdate ?deathdate ?sexLabel ?pseudonym ?occupationLabel WHERE {{ 
          ?autor wdt:P214 "{viaf}" ;
          optional {{ ?autor wdt:P19 ?birthplace . }}
          optional {{ ?autor wdt:P569 ?birthdate . }}
          optional {{ ?autor wdt:P570 ?deathdate . }}
          optional {{ ?autor wdt:P20 ?deathplace . }}
          optional {{ ?autor wdt:P21 ?sex . }}
          optional {{ ?autor wdt:P106 ?occupation . }}
          optional {{ ?autor wdt:P742 ?pseudonym . }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pl". }}}}"""    
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.json_normalize(results['results']['bindings'])
        columns = [e for e in results_df.columns.tolist() if 'value' in e]
        results_df = results_df[results_df.columns.intersection(columns)]       
        for column in results_df.drop(columns='autor.value'):
            results_df[column] = results_df.groupby('autor.value')[column].transform(lambda x: '❦'.join(x.drop_duplicates().astype(str)))
        results_df = results_df.drop_duplicates().reset_index(drop=True)   
        result = results_df.to_dict('records')
        nodegoat_people_df.at[i, 'wikidata'] = result
    except (KeyError, TypeError):
        nodegoat_people_df.at[i, 'wikidata'] = np.nan
    except (HTTPError, RemoteDisconnected) as error:
        print(error)
        time.sleep(61)
        continue
    
    
nodegoat_viaf_wikidata_df = nodegoat_people_df.groupby('samizdatID').filter(lambda x: len(x) == 1)
nodegoat_viaf_wikidata_df = nodegoat_viaf_wikidata_df[nodegoat_viaf_wikidata_df['wikidata'].notnull()].drop(columns=['viaf id', 'viaf score']) 

nodegoat_people_sheet_new = gc.create(f"nodegoat_people_{year}-{month}-{day}", '1UdglvjjX4r2Hzh5BIAr8FjPuWV89NVs6')
worksheet = nodegoat_people_sheet_new.get_worksheet(0)
worksheet.update_title('wikidata ok')

set_with_dataframe(worksheet, nodegoat_viaf_wikidata_df)
nodegoat_people_sheet_new.batch_update({
    "requests": [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet._properties['sheetId'],
                    "dimension": "ROWS",
                    "startIndex": 0,
                    #"endIndex": 100
                },
                "properties": {
                    "pixelSize": 20
                },
                "fields": "pixelSize"
            }
        }
    ]
})

worksheet.freeze(rows=1)
worksheet.set_basic_filter()

nodegoat_people_df = nodegoat_people_df[~nodegoat_people_df['samizdatID'].isin(nodegoat_viaf_wikidata_df['samizdatID'])].drop(columns=['viaf id', 'viaf score'])
nodegoat_people_df = nodegoat_people_df[['samizdatID', 'samizdat_name', 'viaf', 'wikidata']]
#%% szukanie po tytułach książek w danych z BN
tytuly_bn = [file['id'] for file in file_list if file['title'] == 'samizdat_kartoteka_osób'][0]
tytuly_bn_sheet = gc.open_by_key(tytuly_bn)
tytuly_bn_sheet.worksheets()

tytuly_bn_df = get_as_dataframe(tytuly_bn_sheet.worksheet('bn_books'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1)

nodegoat_people_df_location = get_as_dataframe(nodegoat_people_sheet.worksheet('Arkusz1'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()[['Project_ID', 'name_form_id']]

nodegoat_people_df = nodegoat_people_df.merge(nodegoat_people_df_location, left_on='samizdatID', right_on='Project_ID', how='left').drop(columns='Project_ID')

def marc_parser_dict_for_field(string, subfield_code):
    subfield_list = re.findall(f'{subfield_code}.', string)
    dictionary_field = {}
    for subfield in subfield_list:
        subfield_escape = re.escape(subfield)
        string = re.sub(f'({subfield_escape})', r'❦\1', string)
    for subfield in subfield_list:
        subfield_escape = re.escape(subfield)
        regex = f'(^)(.*?\❦{subfield_escape}|)(.*?)(\,{{0,1}})((\❦{subfield_code})(.*)|$)'
        value = re.sub(regex, r'\3', string).strip()
        dictionary_field[subfield] = value
    return dictionary_field

test = copy.deepcopy(nodegoat_people_df)

test = gc.open_by_key('1VNzdJzCaQzPYKLySUD3Jc5D8U-B4c_T2SE68ybc9c9o')
test = get_as_dataframe(test.worksheet('po tytułach'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()


for i, row in tqdm(test.iterrows(), total=test.shape[0]):
    i = 9
    row = test.iloc[i,:]
    locations = row['name_form_id'].split('|')
    try:
        viaf = row['viaf'][1]
        viaf = '101882692'
        list_for_titles = []
        for location in locations:
            location = locations[0]
            location = location.split('-')
            if location[0] == 'bn_books':
                title = tytuly_bn_df[tytuly_bn_df['id'] == int(location[1])][200].reset_index(drop=True)[0]
                if '%e' in title:
                    title = marc_parser_dict_for_field(title, '\%')['%a'] + ' ' + marc_parser_dict_for_field(title, '\%')['%e']
                else:
                    title = marc_parser_dict_for_field(title, '\%')['%a']
                # url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                url = f"https://viaf.org/viaf/search?query=cql.any%20all%20'{title}'&sortKeys=holdingscount&httpAccept=application/json"
                response = requests.get(url)
                response.encoding = 'UTF-8'
                try:
                    samizdat_json = response.json()  
                    samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                    samizdat_json = [[e['record']['recordData']['mainHeadings']['data'], e['record']['recordData']['viafID'], e['record']['recordData']['titles']['work'], len(e['record']['recordData']['sources']['source'])] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
                    try:
                        samizdat_json_selected = [e for e in samizdat_json if e[1] == viaf][0]
                        if type(samizdat_json_selected[0]) == list:
                            nazwa = [e['text'] for e in samizdat_json_selected[0]]
                        elif type(samizdat_json_selected[0]) == dict:
                            nazwa = samizdat_json_selected[0]['text']
                        
                        if type(samizdat_json_selected[2]) == list:
                            tytuly = [e['title'] for e in samizdat_json_selected[2]]
                        elif type(samizdat_json_selected[2]) == dict:
                            tytuly = samizdat_json_selected[2]['title']
                            
                        samizdat_dict = str({'viaf':viaf,'nazwa':nazwa, 'tytuły':tytuly})
                        list_for_titles.append(samizdat_dict)
                        #test.at[i, 'odpytanie po tytułach'] = samizdat_dict
                    except IndexError:
                        test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
                        pass
                except (KeyError, ValueError):
                    pass
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
            list_for_titles = list(set(list_for_titles))
            if list_for_titles:
                test.at[i, 'odpytanie po tytułach'] = list_for_titles
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do usunięcia'
    except TypeError:
        list_for_titles = []
        for location in locations:
            #location = locations[0]
            location = location.split('-')
            if location[0] == 'bn_books':
                title = tytuly_bn_df[tytuly_bn_df['id'] == int(location[1])][200].reset_index(drop=True)[0]
                if '%e' in title:
                    title = marc_parser_dict_for_field(title, '\%')['%a'] + ' ' + marc_parser_dict_for_field(title, '\%')['%e']
                else:
                    title = marc_parser_dict_for_field(title, '\%')['%a']
                url = f"http://www.viaf.org//viaf/search?query=cql.any+=+{title}&maximumRecords=1000&httpAccept=application/json"
                response = requests.get(url)
                response.encoding = 'UTF-8'
                try:
                    samizdat_json = response.json()  
                    samizdat_json = samizdat_json['searchRetrieveResponse']['records']
                    samizdat_json = [[e['record']['recordData']['mainHeadings']['data'], e['record']['recordData']['viafID'], e['record']['recordData']['titles']['work'], len(e['record']['recordData']['sources']['source'])] for e in samizdat_json if e['record']['recordData']['nameType'] == 'Personal']
                    if len(samizdat_json) == 1:
                        samizdat_json_selected = samizdat_json[0]
                        if type(samizdat_json_selected[0]) == list:
                            nazwa = [e['text'] for e in samizdat_json_selected[0]]
                        elif type(samizdat_json_selected[0]) == dict:
                            nazwa = samizdat_json_selected[0]['text']
                        
                        if type(samizdat_json_selected[2]) == list:
                            tytuly = [e['title'] for e in samizdat_json_selected[2]]
                        elif type(samizdat_json_selected[2]) == dict:
                            tytuly = samizdat_json_selected[2]['title']
                            
                        samizdat_dict = str({'viaf':viaf,'nazwa':nazwa, 'tytuły':tytuly})
                        list_for_titles.append(samizdat_dict)
                    else:
                        test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
                        pass
                except (KeyError, ValueError, TypeError):
                    pass
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do sprawdzenia'
            list_for_titles = list(set(list_for_titles))
            if list_for_titles:
                test.at[i, 'odpytanie po tytułach'] = list_for_titles
            else:
                test.at[i, 'odpytanie po tytułach'] = 'wiersz do usunięcia'

try:
    set_with_dataframe(nodegoat_people_sheet_new.worksheet('po tytułach'), test)
except gs.WorksheetNotFound:
    nodegoat_people_sheet_new.add_worksheet(title="po tytułach", rows="100", cols="20")
    set_with_dataframe(nodegoat_people_sheet_new.worksheet('po tytułach'), test)
    
worksheet = nodegoat_people_sheet_new.worksheet('po tytułach')

nodegoat_people_sheet_new.batch_update({
    "requests": [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet._properties['sheetId'],
                    "dimension": "ROWS",
                    "startIndex": 0,
                    #"endIndex": 100
                },
                "properties": {
                    "pixelSize": 20
                },
                "fields": "pixelSize"
            }
        }
    ]
})

worksheet.freeze(rows=1)
worksheet.set_basic_filter()
 #%%               
                
#wyszukiwanie osób przez ten link:
# http://viaf.org/viaf/search?query=local.names%20all%20%22Babi%C5%84ski%20Stanis%C5%82aw%22&sortKeys=holdingscount&httpAccept=application/json            
   
#%% wgranie danych dla PW 14.04
nodegoat_people = [file['id'] for file in file_list if file['title'] == 'nodegoat_people_2021-04-07'][0]
nodegoat_people_sheet = gc.open_by_key(nodegoat_people)
nodegoat_people_sheet.worksheets()

nodegoat_people_df = get_as_dataframe(nodegoat_people_sheet.worksheet('po tytułach'), evaluate_formulas=True).dropna(how='all').dropna(how='all', axis=1).drop_duplicates()

df_grouped = nodegoat_people_df.groupby('samizdatID')
df_ok_z_viaf = pd.DataFrame()
df_ok_bez_viaf = pd.DataFrame()
df_reszta_z_trafieniami = pd.DataFrame()
df_reszta_bez_trafienia = pd.DataFrame()
for name, group in tqdm(df_grouped, total=len(df_grouped)):
    try:
        wiersze_do_usuniecia = group['odpytanie po tytułach'].value_counts()['wiersz do usunięcia']
    except KeyError:
        wiersze_do_usuniecia = 0
    if group.shape[0] == wiersze_do_usuniecia:
        df_reszta_bez_trafienia = df_reszta_bez_trafienia.append(group)
    else:
        df = group[group['odpytanie po tytułach'] != 'wiersz do usunięcia']
        jest_viaf = any(df['viaf'].notnull())
        if df.shape[0] == 1 and jest_viaf == True:
            df_ok_z_viaf = df_ok_z_viaf.append(df)
        elif df.shape[0] == 1 and jest_viaf == False:
            df_ok_bez_viaf = df_ok_bez_viaf.append(df)
        else:
            df_reszta_z_trafieniami = df_reszta_z_trafieniami.append(df)
    
arkusze = ['ok z viaf', 'ok bez viaf', 'reszta z trafieniami', 'reszta bez trafienia']
dfs = [df_ok_z_viaf, df_ok_bez_viaf, df_reszta_z_trafieniami, df_reszta_bez_trafienia]
for arkusz, dataf in zip(arkusze, dfs):
    try:
        set_with_dataframe(nodegoat_people_sheet.worksheet(arkusz), dataf)
    except gs.WorksheetNotFound:
        nodegoat_people_sheet.add_worksheet(title=arkusz, rows="100", cols="20")
        set_with_dataframe(nodegoat_people_sheet.worksheet(arkusz), dataf)















