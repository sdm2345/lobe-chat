import { AzureOpenAI } from 'openai';
import type {  ChatCompletionMessageParam } from 'openai/resources/chat/completions';

import { LobeRuntimeAI } from '../BaseAI';
import { AgentRuntimeErrorType } from '../error';
import { ChatCompetitionOptions, ChatStreamPayload, ModelProvider } from '../types';
import { AgentRuntimeError } from '../utils/createError';
import { debugStream } from '../utils/debugStream';
import { StreamingResponse } from '../utils/response';
import {AzureOpenAIStream} from "@/libs/agent-runtime/utils/streams";

export class LobeAzureOpenAI implements LobeRuntimeAI {
  client: AzureOpenAI;
  baseURL: string;

  constructor(endpoint?: string, apikey?: string, apiVersion?: string) {
    if (!apikey || !endpoint)
      throw AgentRuntimeError.createError(AgentRuntimeErrorType.InvalidProviderAPIKey);

    console.log(`init AzureOpenAI endpoint:${endpoint}`);

    this.client = new AzureOpenAI({
      apiKey: apikey,

      defaultQuery: { 'api-version': apiVersion || '2024-02-15-preview' },
      endpoint,
    });

    this.baseURL = endpoint;
  }

  private createStreamingResponse(content: string, id: string) {
    return new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        const lines = content.split('\n');
        for (const line of lines) {
          const msg = `id: ${id}\nevent: text\ndata: ${JSON.stringify(line + '\n')}\n\n`;
          controller.enqueue(encoder.encode(msg));
        }
        controller.close();
      }
    });
  }

  async chat(payload: ChatStreamPayload, options?: ChatCompetitionOptions) {
    const { messages, model, ...params } = payload;
    // @ts-ignore
    delete params['stop'];
    // mock stream api for o3-mini and o1-preview models
    if (model.slice(0, 8) === 'o3-mini-' || model === 'o1-mini' || model === 'o1-preview') {
      const params2:Record<string, any> = { ...params };
      delete params2['stream'];
      delete params2['temperature'];
      const messages2 = messages.map((item) => {
        if(item.role === 'system') {
          // @ts-ignore
          item.role = 'user';
        }
        return item
    });

      let modelNew = model;
      let reasoning_effort = 'low';

      if (model.slice(0, 8) === 'o3-mini-') {
        modelNew = 'o3-mini';
        reasoning_effort = model.slice(8);
        const TypeMap: Record<string, string> = {
          'high': 'high',
          'low': 'low',
          'medium': 'medium'
        };
        reasoning_effort = TypeMap[reasoning_effort] || 'low';
        params2['reasoning_effort'] = reasoning_effort;
      }

      try {
        const response = await this.client.chat.completions.create({
          messages: messages2,
          model: modelNew,
          ...params2,
        });

        const content = response.choices[0]?.message?.content || '';
        return StreamingResponse(this.createStreamingResponse(content, response.id));
      } catch (error) {
        throw this.handleError(error);
      }
    }

    try {
      const response = await this.client.chat.completions.create({
        messages: messages as ChatCompletionMessageParam[],
        model,
        stream: true,
        ...params,
      });

      const [debug, prod] = response.tee();

      if (process.env.DEBUG_AZURE_CHAT_COMPLETION === '1') {
        // @ts-ignore
        debugStream(debug).catch(console.error);
      }

       return StreamingResponse(AzureOpenAIStream(prod, options?.callback), {
        headers: options?.headers,
      });
    } catch (e) {
      throw this.handleError(e);
    }
  }

  private handleError(e: any) {
    let error = e as { [key: string]: any; code: string; message: string };

    if (error.code) {
      switch (error.code) {
        case 'DeploymentNotFound': {
          error = { ...error, deployId: error.model };
          break;
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
    const regex = /^(https:\/\/)([^.]+)(\.openai\.azure\.com\/.*)$/;
    return url.replace(regex, (match, protocol, subdomain, rest) => {
      return `${protocol}***${rest}`;
    });
  };
}
