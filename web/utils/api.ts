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

export async function getDatasetData(dataset: string) : Promise<DatasetResults> {
  const dataRaw = await fs.readFile(`./data/${dataset}.json`, {encoding: "utf8"});
  let data: DatasetResults = JSON.parse(dataRaw);
  const defaultMetric = data.default_metric || data.metrics[0].id;
  const sign = data.metrics.find((m: Metric) => m.id === defaultMetric)?.ascending?-1:1;
  data.methods.sort((a: any, b: any) => sign * (a[defaultMetric]-b[defaultMetric]));
  return data;
}

export async function getMethodData(method: string) : Promise<DatasetResults[]> {
  let datasets: DatasetResults[] = [];
  const items = await fs.readdir("./data");
  for (let i = 0; i < items.length; i++) {
    const datasetId = items[i].replace(".json", "");
    let data = await getDatasetData(datasetId);
    data.methods = data.methods.filter((m: MethodResults) => m.id === method);
    datasets.push(data);
  }
  return datasets;
}

export async function getDatasets() : Promise<Dataset[]> {
  let datasets: Dataset[] = [];
  const items = await fs.readdir("./data");
  for (let i = 0; i < items.length; i++) {
    const datasetId = items[i].replace(".json", "");
    const dataset = await getDatasetData(datasetId);
    datasets.push({
      id: datasetId,
      name: dataset.name,
    });
  }
  return datasets;
}

export async function getMethods() : Promise<BaseInfo[]> {
  let methods: BaseInfo[] = [];
  const methodIds = new Set<string>();
  const items = await fs.readdir("./data");
  for (let i = 0; i < items.length; i++) {
    const datasetId = items[i].replace(".json", "");
    const dataset = await getDatasetData(datasetId);
    for (let method of dataset.methods) {
      if (!methodIds.has(method.id)) {
        methodIds.add(method.id);
        methods.push({
          id: method.id,
          name: method.name,
          description: method.description,
          link: method.link,
          paper_title: method.paper_title,
          paper_link: method.paper_link,
          paper_authors: method.paper_authors,
        });
      }
    }
  }
  return methods;
}