import readlineSync from 'readline-sync';
import * as dotenv from 'dotenv';
dotenv.config();

import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';

const embeddings = new HuggingFaceTransformersEmbeddings({
  model: 'Xenova/all-MiniLM-L6-v2',
});

function getPineconeClient() {
  const apiKey = process.env.PINECONE_API_KEY;
  const indexName = process.env.PINECONE_INDEX_NAME;
  const controllerHostUrl = process.env.PINECONE_CONTROLLER_HOST;

  if (!apiKey) {
    throw new Error('PINECONE_API_KEY is missing in .env');
  }
  if (!indexName) {
    throw new Error('PINECONE_INDEX_NAME is missing in .env');
  }

  console.log('Using Pinecone config:', { indexName, controllerHostUrl });
  const config = { apiKey };
  if (controllerHostUrl) config.controllerHostUrl = controllerHostUrl;
  return new Pinecone(config);
}

async function storeAnswer(question, answer, pineconeIndex) {
  const vector = await embeddings.embedQuery(`${question}\n${answer}`);
  const id = `qa-${Date.now()}`;
  await pineconeIndex.upsert([
    {
      id,
      values: vector,
      metadata: {
        question,
        answer,
        createdAt: new Date().toISOString(),
      },
    },
  ]);
  console.log(`Stored answer in Pinecone record ${id}`);
}

async function chatting(question) {
  const pinecone = getPineconeClient();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
  });

  const result = await vectorStore.similaritySearch(question, 5);

  const onlyText = result.map((doc, index) => ({
    rank: index + 1,
    text: doc.pageContent,
  }));

  console.log('\nPrinting top 5 Documents');
  onlyText.forEach((elem) => {
    console.log(`Rank ${elem.rank}`);
    console.log(elem.text);
  });

  const answer = onlyText.map((doc) => `Rank ${doc.rank}: ${doc.text}`).join('\n\n');
  await storeAnswer(question, answer, pineconeIndex);

  return onlyText;
}

async function main() {
  const question = readlineSync.question('Ask me Anything about SQL: ');
  const response = await chatting(question);
  console.log('\nFinal Response:');
  console.log(response);

  main();
}

main();