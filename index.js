import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import * as dotenv from "dotenv";
dotenv.config();
import {Pinecone} from "@pinecone-database/pinecone";
import {PineconeStore} from "@langchain/pinecone";

import { HuggingFaceTransformersEmbeddings } from '@langchain/community/embeddings/hf_transformers';
//load environment variables from .env file
async function indexDocuments() {
    const PDF_PATH = "./book.pdf";
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log("Documents loaded.");
// chunking setup
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 20,
    });
    const chunkDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("chunking completed.");
// embeddings setup
const embeddings = new HuggingFaceTransformersEmbeddings({
    model: 'Xenova/all-MiniLM-L6-v2',
}
);
console.log("Embeddings model loaded.");
const test = await embeddings.embedQuery("Hello world");
console.log(test);
console.log(test.length);

//pinecone setup
if (!process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX_NAME) {
  throw new Error("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in .env");
}

const pineconeConfig = {
  apiKey: process.env.PINECONE_API_KEY,
};
if (process.env.PINECONE_CONTROLLER_HOST) {
  pineconeConfig.controllerHostUrl = process.env.PINECONE_CONTROLLER_HOST;
}

const pinecone = new Pinecone(pineconeConfig);
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
console.log("Pinecone index initialized:", { indexName: process.env.PINECONE_INDEX_NAME, controllerHostUrl: pineconeConfig.controllerHostUrl });

await PineconeStore.fromDocuments(chunkDocs, embeddings, {
  pineconeIndex,
  maxConcurrency: 4,
});
console.log("Documents indexed.");

}
indexDocuments();