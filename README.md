# 北海道のコロナウィルス感染者数を過去14日間の感染者数と曜日から予想するプログラム
## 使い方
GitHubからコードをクローンする
```
$ git clone https://github.com/kawata-yuya/Hokkaido-Coronavirus-Positive-Regression-AI
$ cd Hokkaido-Coronavirus-Positive-Regression-AI
``` 
必要なライブラリーのダウンロード
```
$ pip3 install -r requirements.txt
```
予想するために必要なパラメーターを設定
main_inference.pyのmain()の下にある[User Settings](https://github.com/kawata-yuya/Hokkaido-Coronavirus-Positive-Regression-AI/blob/master/main_inference.py#L15)に予想したい日の日付、過去14日間の感染者数と曜日を入力。  
ファイルの実行
```
$ python3 main_inference.py  
```
実行すると、北海道コロナウィルス感染者数の計算予想結果と今後の予想グラフが表示されます。  

## 注意  
計算結果はあくまでも予想であり、本当の値を示すわけではありません。  
また、作成者はこの北海道コロナウィルス感染者数の計算予想結果に一切責任を負いません。

## License
MIT