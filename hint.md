             pages.jsonl
                  │
                  ▼
        TF-IDF всех статей
                  │
                  ├──────────────────────────┐
                  │                          │
          history.jsonl                 Запрос пользователя
                  │                          │
                  ▼                          ▼
       Профиль пользователя          Intent classifier
                  │                          │
                  └──────────┬───────────────┘
                             ▼
                    PersonalizedRetriever
                             │
           ┌─────────────────┼──────────────────┐
           ▼                 ▼                  ▼
     query_score       profile_score      category_score
           └─────────────────┼──────────────────┘
                             ▼
                     итоговый ranking
                             ▼
                      лучшие статьи
                             ▼
                         Generator
                             ▼
                           Ответ