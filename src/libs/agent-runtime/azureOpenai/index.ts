import {AzureKeyCredential, ChatRequestMessage, GetChatCompletionsOptions, OpenAIClient,} from '@azure/openai';

import {LobeRuntimeAI} from '../BaseAI';
import {AgentRuntimeErrorType} from '../error';
import {ChatCompetitionOptions, ChatStreamPayload, ModelProvider} from '../types';
import {AgentRuntimeError} from '../utils/createError';
import {debugStream} from '../utils/debugStream';
import {StreamingResponse} from '../utils/response';
import {AzureOpenAIStream} from '../utils/streams';

export class LobeAzureOpenAI implements LobeRuntimeAI {
  client: OpenAIClient;

  constructor(endpoint?: string, apikey?: string, apiVersion?: string) {
    if (!apikey || !endpoint)
      throw AgentRuntimeError.createError(AgentRuntimeErrorType.InvalidProviderAPIKey);

    this.client = new OpenAIClient(endpoint, new AzureKeyCredential(apikey), {apiVersion});

    this.baseURL = endpoint;
  }

  baseURL: string;

  async chat(payload: ChatStreamPayload, options?: ChatCompetitionOptions) {
    // ============  1. preprocess messages   ============ //
    const camelCasePayload = this.camelCaseKeys(payload);
    const {messages, model, maxTokens = 2048, ...params} = camelCasePayload;

    // ============  2. send api   ============ //

    // mock stream api for o1-mini and o1-preview
    if (model === 'o1-mini' || model === 'o1-preview') {
      const params2 = {...params}
      delete params2['stream']
      delete params2['temperature']
      const messages2 = messages.map((item:ChatRequestMessage) => {
        item.role = item.role === 'system' ? 'user' : item.role;
        return item;
      })
      const response = await this.client.getChatCompletions(
        model,
        messages2 as ChatRequestMessage[],
        {...params2, abortSignal: options?.signal} as GetChatCompletionsOptions,
      );

      const sendMessage = function (ctl: ReadableStreamDefaultController, evt: {
        data: any,
        event: string,
        id: string,
      }) {
        const encoder: TextEncoder = new TextEncoder();
        const msg = `id: ${evt.id}
event: ${evt.event}
data: ${JSON.stringify(evt.data)}\n\n`
        ctl.enqueue(encoder.encode(msg));
      }

      return StreamingResponse(new ReadableStream({
        start: (controller) => {

          let content = response.choices[0].message?.content
          if(content){
            const arr = content.split('\n')
            for(const line of arr) {
              sendMessage(controller, {
                  data: line+'\n',
                  event: 'text',
                  id: response.id
                }
              )
            }
          }
          controller.close()
        }
      }));
    }
    try {
      const response = await this.client.streamChatCompletions(
        model,
        messages as ChatRequestMessage[],
        {...params, abortSignal: options?.signal, maxTokens} as GetChatCompletionsOptions,
      );

      const [debug, prod] = response.tee();

      if (process.env.DEBUG_AZURE_CHAT_COMPLETION === '1') {
        debugStream(debug).catch(console.error);
      }

      return StreamingResponse(AzureOpenAIStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      let error = e as { [key: string]: any; code: string; message: string };

      if (error.code) {
        switch (error.code) {
          case 'DeploymentNotFound': {
            error = {...error, deployId: model};
          }
        }
      } else {
        error = {
          cause: error.cause,
          message: error.message,
          name: error.name,
        } as any;
      }

      const errorType = error.code
        ? AgentRuntimeErrorType.ProviderBizError
        : AgentRuntimeErrorType.AgentRuntimeError;

      throw AgentRuntimeError.chat({
        endpoint: this.maskSensitiveUrl(this.baseURL),
        error,
        errorType,
        provider: ModelProvider.Azure,
      });
    }
  }

  // Convert object keys to camel case, copy from `@azure/openai` in `node_modules/@azure/openai/dist/index.cjs`
  private camelCaseKeys = (obj: any): any => {
    if (typeof obj !== 'object' || !obj) return obj;
    if (Array.isArray(obj)) {
      return obj.map((v) => this.camelCaseKeys(v));
    } else {
      for (const key of Object.keys(obj)) {
        const value = obj[key];
        const newKey = this.tocamelCase(key);
        if (newKey !== key) {
          delete obj[key];
        }
        obj[newKey] = typeof obj[newKey] === 'object' ? this.camelCaseKeys(value) : value;
      }
      return obj;
    }
  };

  private tocamelCase = (str: string) => {
    return str
      .toLowerCase()
      .replaceAll(/(_[a-z])/g, (group) => group.toUpperCase().replace('_', ''));
  };

  private maskSensitiveUrl = (url: string) => {
    // 使用正则表达式匹配 'https://' 后面和 '.openai.azure.com/' 前面的内容
    const regex = /^(https:\/\/)([^.]+)(\.openai\.azure\.com\/.*)$/;

    // 使用替换函数
    return url.replace(regex, (match, protocol, subdomain, rest) => {
      // 将子域名替换为 '***'
      return `${protocol}***${rest}`;
    });
  };
}
