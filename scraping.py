## 歌詞スクレイピング

# #ライブラリインポート
import requests
from bs4 import BeautifulSoup
import pandas as pd

#取得したデータを格納するデータフレームを作成
df = pd.DataFrame(columns=['artist','title','text'])

#Uta-Net先頭URL
base_url = 'https://www.uta-net.com'
#King Gnuの歌詞一覧ページURL
artist_list = [{"name":"King Gnu", "url":'https://www.uta-net.com/artist/23343/'},
        {"name":"Official髭男dism", "url":'https://www.uta-net.com/artist/18093/'},
        {"name":"米津玄師", "url":'https://www.uta-net.com/artist/12795/'},
        {"name":"back number", "url":'https://www.uta-net.com/artist/8613/'},
        {"name":"Mrs. GREEN APPLE", "url":"https://www.uta-net.com/artist/18526/"}
        ]

for i in range(0, len(artist_list)):
  #歌詞一覧ページのHTML取得
  response = requests.get(artist_list[i]["url"])
  soup = BeautifulSoup(response.text, 'lxml')
  links = soup.find_all('td', class_='sp-w-100 pt-0 pt-lg-2')
  #歌詞ページより、情報を取得
  for link in links:
    a = base_url + (link.a.get('href'))

    #歌詞ページよりHTMLを取得
    response_a = requests.get(a)
    soup_a = BeautifulSoup(response_a.text, 'lxml')
    #アーティスト名取得
    # artist = soup_a.find('h3').text.replace('\n','')
    #アーティストラベル
    artist = i
    #title取得
    title = soup_a.find('h2').text
    #歌詞取得
    text = soup_a.find('div', itemprop='lyrics').text.replace('\n','')
    text = text.replace('この歌詞をマイ歌ネットに登録 >このアーティストをマイ歌ネットに登録 >','')

    #取得したデータフレームに追加
    temp_df = pd.DataFrame([[artist],[title],[text]], index=df.columns).T
    df = df.append(temp_df, ignore_index=True)

df.to_csv('df.tsv', sep='\t', mode="w")