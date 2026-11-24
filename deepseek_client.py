"""Official DeepSeek API adapter. No Stack transport or Stack credential fallback."""
import json,os,urllib.request,urllib.error

class MissingDeepSeekKey(RuntimeError):pass

def config():
    return {
        'provider': 'deepseek_official',
        'base_url': 'https://api.deepseek.com',
        'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
        'thinking': {'type': 'disabled'},
        'timeout_seconds': 180,
    }


def credential():
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    raise MissingDeepSeekKey('请在本机设置 DEEPSEEK_API_KEY 环境变量。')


class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None

def _request(path,payload=None,timeout=None):
 cfg=config();key=credential()
 headers={'Authorization':'Bearer '+key}
 body=None
 if payload is not None:body=json.dumps(payload).encode();headers['Content-Type']='application/json'
 req=urllib.request.Request(cfg['base_url'].rstrip('/')+path,data=body,headers=headers)
 try:
  with urllib.request.build_opener(NoRedirect()).open(req,timeout=timeout or cfg.get('timeout_seconds',180)) as response:return json.load(response)
 except urllib.error.HTTPError as e:
  label={401:'API Key 无效或未授权',402:'账户余额不足',429:'触发限流'}.get(e.code,'请求失败')
  raise RuntimeError(f'DeepSeek 官网接口 HTTP {e.code}：{label}') from None
 except (urllib.error.URLError,TimeoutError) as e:
  raise RuntimeError(('DeepSeek 官网接口连接失败：'+str(e)).replace(key,'[redacted]')) from None

def catalog():return [item['id'] for item in _request('/models',timeout=30)['data']]

def chat(model,messages,max_tokens=8192,timeout=None,thinking=None):
 cfg=config();mode=thinking or cfg.get('thinking',{'type':'enabled'})
 payload={'model':model or cfg['model'],'messages':messages,'max_tokens':max_tokens,'stream':False,'thinking':mode}
 if mode.get('type')=='enabled':payload['reasoning_effort']=cfg.get('reasoning_effort','high')
 else:payload['temperature']=0.3
 obj=_request('/chat/completions',payload,timeout)
 choice=obj['choices'][0];content=choice['message'].get('content') or ''
 if not content:raise RuntimeError('官网未返回最终正文，结束状态：'+str(choice.get('finish_reason')))
 return content,{'provider':'deepseek_official','model':payload['model'],'usage':obj.get('usage'),'finish_reason':choice.get('finish_reason')}
