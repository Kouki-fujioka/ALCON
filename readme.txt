動作仕様書（Python 3.10）

1. 仮想環境の構築
	・Linux / macOS
	1.a 仮想端末を開く
	1.b 次のコマンドを実行
		$ python3 -m venv venv
		$ source venv/bin/activate
	1.c 仮想環境上に依存ライブラリをインストール
		$ pip install -r requirements.txt

	・Windows
	1.a 仮想端末を開く（PowerShellを推奨，以下はPowerShellでの手順です．）
	1.b 次のコマンドを実行
		PS>  python -m venv venv
		PS> ./venv/Scripts/Activate.ps1
	1.c 仮想環境上に依存ライブラリをインストール
		PS> ./venv/Scripts/Python.exe -m pip install -r requirements.txt

2. 入力ファイルの指定
    inフォルダに検知したい動画および検知のためのcsvファイルを入れ, main.py内で指定

3. プログラムの実行
	・Linux / macOS
	$ python main.py

	・Windows
	PS> ./venv/Scripts/Python.exe main.py

4. 出力
   outフォルダには検知を行ったMP4および検知した魚種ごとのカウント数を示したcsvファイルが出力される
