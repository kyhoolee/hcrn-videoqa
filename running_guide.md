- Connect to server
ssh <username>@pagi.it.deakin.edu.au -L 5010:localhost:5010

- Access demo page
http://localhost:5010/videoqa/

- Demo for TGIF-QA action 
Input: video gif, question, 5-choice list, answer-index
Output: json result from model

- Next demo: 

+ Update UI: 
    waiting/loading UI when processing --> show processing steps smilar to VQA demo --> need to create websocket API
    Show video gif and answer instead of showing JSON result 

+ Update model:
    Clone API and UI for other model and other question-type
