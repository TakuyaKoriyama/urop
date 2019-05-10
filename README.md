# readme
jsjsk
- hsdgj
- kjhkjs


## githubにファイルをアップロードする手順

1. ファイルを編集する。たとえば`main.py`の中にrender関数を追加したとする
1. `git add main.py`
1. `git commit -m "added render function"`
1. `git push origin master`

ちなみに、`git status` で状況をいつでも確認できる


## github上の更新を取り込む場合

例えば松井が更新したときにそれをローカルPCに反映させる場合

1. ローカルPC上でプロジェクトの位置に移動する
1. `git pull origin master`

もしエラーが出たら便宜ググッて対応