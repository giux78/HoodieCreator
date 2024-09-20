from upstash_redis import Redis
import os
import json

redis = Redis(url=os.getenv('UPSTASH_REDIS_REST_URL'), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
keys = redis.keys("user:thread:*")

with open('./mii-llm-conversations.jsonl', 'w') as f:
    for key in keys:
        email = key.split(':')[-1]
        for thread in redis.zrange(f'user:thread:{email}', 0, -1):
            #©print(thread)
            threadInfo = redis.hgetall(thread)
            #print(threadInfo)
            if threadInfo:
                assistandId = json.loads(threadInfo['assistantId'])
                #print(assistandId['assistant_id'])
                if redis.exists(f'{thread}:messages') and \
                        assistandId['assistant_id'] == 'asst_zefiro' and \
                        int(threadInfo['created_at']) >= 1715108663: 
                    messages = redis.get(f'{thread}:messages')
                    threadInfo['messages'] = json.loads(messages)
                    print(threadInfo)
                    f.write(json.dumps(threadInfo))
                    f.write('\n')
f.close()