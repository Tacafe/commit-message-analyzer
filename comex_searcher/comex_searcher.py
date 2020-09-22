import PySimpleGUI as sg

#  セクション1 - オプションの設定と標準レイアウト
sg.theme('DarkAmber')

layout = [
    [sg.Text('検索条件')],
    [sg.Text('word/phrase', size=(20, 1)), sg.InputText()],
    [sg.Combo(['この語句と共起するNP/VP', 'この語句を含むNP/VP', 'この語句を含むNP/VPと共起するNP/VP'], size=(30, 1), default_value='この語句と共起するNP/VP')],
    [sg.Text('Top-N', size=(5, 1)), sg.InputText('10')],
    [sg.Text('閾値', size=(5, 1)), sg.InputText('0')],
    [sg.Submit(button_text='Search')]
]

# セクション 2 - ウィンドウの生成
window = sg.Window('CommitMessage EXxpression Searcher', layout)

# セクション 3 - イベントループ
while True:
    event, values = window.read()

    if event is None:
        print('exit')
        break

    if event == '実行ボタン':
        show_message = "名前：" + values[0] + 'が入力されました。\n'
        show_message += "住所：" + values[1] + 'が入力されました。\n'
        show_message += "電話番号：" + values[2] + "が入力されました。"
        print(show_message)

        # ポップアップ
        sg.popup(show_message)

# セクション 4 - ウィンドウの破棄と終了
window.close()
