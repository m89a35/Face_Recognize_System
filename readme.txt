實時抓臉程式 : 
    face_record_simplify.py // 最陽春版本
    執行後 會直接開始抓臉 
    要設置張數 可到num做調整

    face_record.py // 可控制何時開始版本 
    執行後 輸入名稱 按下r 抓臉的框線會變綠色 收集滿300張後會結束。
    提早退出 按下q 

    face_record_show_landmark.py // 影片版本
    改成影片方式，要去程式碼內修改人名。
    若沒有影片 會報錯 rgb什麼的 表示沒抓到影片 要確定影片在該在的位置 或是更改程式碼。
    一樣抓完num數會結束。

    face_record_with_face_align.py // 影片+會矯正臉部的版本
    一樣用影片，但會將臉部進行轉正。

訓練程式 :



辨識程式 :
    face_detect_recog.py // 陽春版
    1.image的部分有錯  執行會顯示 frame 沒被定義 是因為之前寫的時候  frame 跟 image兩個搞錯了
        要到imgae的程式碼裡面 把frame 改成image 就可以了。
    2.出現 RuntimeError: Unsupported image type, must be 8bit gray or RGB image.
        影片有時候路徑只有一個\ 會有問題  寫兩個就可改善。
        出現這個error 都是 沒抓到影片或圖片的問題。
    3. cam沒問題 只是出現的有點慢而已。

    face_detect_recog_simplify.py // 程式碼演進過程而已 不用看

    face_detect_recog_check_version.py // 加了點名的版本。
        但我沒有用到點名的東西 所以就略過不執行。

    face_detect_recog_with_face_align.py // 加了 face align的版本
        可以正常執行，可是會出現兩個視窗 然後另一個視窗有點不知所云。





安裝時會遇到的問題 
    1.ImportError: cannot import name 'joblib'
        把所有要使用joblib的地方 不要從sklearn import 直接改import joblib 
        因為現在sklearn已經沒支援joblib了 那是舊套件
    


執行時遇到的問題
    1.unable to open XXX
        可以試看看路徑都不要有中文出現。


