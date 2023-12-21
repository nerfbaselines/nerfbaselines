import { promises as fs } from "fs";

export interface BaseInfo {
  id: string;
  name: string;
  description?: string;
  link?: string;
  paper_title?: string;
  paper_link?: string;
  paper_authors?: string[];
}

export interface MethodResults extends BaseInfo {
}

export interface Metric extends BaseInfo {
  ascending: boolean;
}

export interface Scene {
  id: string;
  name?: string;
}

export interface DatasetResults extends BaseInfo {
  methods: MethodResults[];
  metrics: Metric[];
  scenes: Scene[];
  default_metric?: string;
}

export interface Dataset {
  name: string;
  id: string;
}

const datasetOrder = ["mipnerf360", "blender"];

export async function getDatasetData(dataset: string) : Promise<DatasetResults> {
  const dataRaw = await fs.readFile(`./data/${dataset}.json`, {encoding: "utf8"});
  let data: DatasetResults = JSON.parse(dataRaw);
  const defaultMetric = data.default_metric || data.metrics[0].id;
  const sign = data.metrics.find((m: Metric) => m.id === defaultMetric)?.ascending?-1:1;
  data.methods.sort((a: any, b: any) => sign * (a[defaultMetric]-b[defaultMetric]));
  return data;
}

async function getAllDatasets() : Promise<DatasetResults[]> {
  let datasetIds = ((await fs.readdir("./data"))
    .filter((f: string) => f.endsWith(".json"))
    .map((f: string) => f.replace(".json", ""))
  );
  datasetIds = (datasetOrder
    .filter((d: string) => datasetIds.includes(d))
    .concat(datasetIds.filter((d: string) => !datasetOrder.includes(d)))
  );

  return await Promise.all(datasetIds.map((datasetId: string) => getDatasetData(datasetId)));
}

export async function getDatasets() : Promise<Dataset[]> {
  return (await getAllDatasets()).map((d: DatasetResults) => ({
    id: d.id,
    name: d.name
  }));
}

export async function getMethodData(method: string) : Promise<DatasetResults[]> {
  return (await getAllDatasets()).map((d: DatasetResults) => ({
    ...d,
    methods: d.methods.filter((m: MethodResults) => m.id === method)
   })).filter((d: DatasetResults) => d.methods.length > 0);
}

export async function getMethods() : Promise<BaseInfo[]> {
  const datasets = await getAllDatasets();
  let methodIds = new Set<string>();
  let rawMethods = datasets.flatMap((d: DatasetResults) => d.methods).filter((m: MethodResults) => {
    const out = !methodIds.has(m.id)
    methodIds.add(m.id);
    return out;
  });
  return rawMethods.map((method: MethodResults) => ({
    id: method.id,
    name: method.name,
    description: method.description,
    link: method.link,
    paper_title: method.paper_title,
    paper_link: method.paper_link,
    paper_authors: method.paper_authors,
  }));
}