import { IconArrowBackUp } from '@tabler/icons-react';
import DatasetResultsTable from '@/app/components/DatasetResultsTable';
import Link from 'next/link';
import * as api from '@/utils/api';


export async function generateStaticParams() {
  const datasets = await api.getDatasets();
  const methods = await api.getMethods();
  return datasets.map((dataset) => ({
    slug: dataset.id.replace(":", "--"),
  })).concat(methods.map((method) => ({
    slug: method.id.replace(":", "--"),
  })));
}


async function DatasetResults(dataset: string) {
  const datasetResults = await api.getDatasetData(dataset);
  return (
    <main>
      <div className="results-page__header">
        <h1>
          {datasetResults.name}<Link className="text-blue" href="/"><IconArrowBackUp size={24} /></Link>
        </h1>
        <p className="text-justify">
          {datasetResults.description}
        </p>
        {(datasetResults.link || datasetResults.paper_link) && (
          <div className="results-page__header-links">
            {datasetResults.link && (
              <div>Web: <a href={datasetResults.link} className="text-blue link-underline">{datasetResults.link}</a></div>
            )}
            {datasetResults.paper_link && (
              <div>Paper: <a href={datasetResults.paper_link} className="text-blue link-underline">{datasetResults.paper_link}</a></div>
            )}
          </div>
        )}
      </div>
      <DatasetResultsTable datasetId={dataset} methodResults={datasetResults.methods} metrics={datasetResults.metrics} scenes={datasetResults.scenes} />
      {datasetResults.metrics.map((metric) => (
        <div key={metric.id} className="result-set-panel">
          <div className="result-set-panel__header">
            <h2>{metric.name}</h2>
            <p className="text-justify">{metric.description}</p>
          </div>
          <DatasetResultsTable 
            datasetId={dataset} 
            methodResults={datasetResults.methods} 
            metricDetail={metric.id} 
            metrics={datasetResults.metrics}
            scenes={datasetResults.scenes} />
        </div>
      ))}
    </main>
  );
}

async function MethodResults(method: string) {
  const allDatasetResults = await api.getMethodData(method);
  const methodInfo = allDatasetResults[0].methods[0];
  return (
    <main className="main-container">
      <div className="results-page__header">
        <h1>
          {methodInfo.name}<Link className="text-blue" href="/"><IconArrowBackUp size={24} /></Link>
        </h1>
        <p className="text-justify">
          {methodInfo.description}
        </p>
        {(methodInfo.link || methodInfo.paper_link) && (
          <div className="results-page__header-links">
            {methodInfo.link && (
              <div>Web: <a href={methodInfo.link} className="text-blue link-underline">{methodInfo.link}</a></div>
            )}
            {methodInfo.paper_link && (
              <div>Paper: <a href={methodInfo.paper_link} className="text-blue link-underline">{methodInfo.paper_link}</a></div>
            )}
          </div>
        )}
      </div>
      {allDatasetResults.map((datasetResults) => (
        <div key={datasetResults.id} className="result-set-panel">
          <div className="result-set-panel__header">
            <h2>
              {datasetResults.name}
            </h2>
            <p className="text-justify">{datasetResults.description}</p>
          </div>
          <DatasetResultsTable 
            datasetId={datasetResults.id} 
            methodResults={datasetResults.methods} 
            metrics={datasetResults.metrics}
            showMethod={false}
            scenes={datasetResults.scenes} />
        </div>
      ))}
    </main>
  );
}


export default async function Page({ params }: {
  params: {
    slug: string
  }
}) {
  const slug = params.slug.replace("--", ":");
  const datasets = await api.getDatasets();
  if (datasets.find((dataset) => dataset.id === slug) !== undefined) {
    return await DatasetResults(slug);
  } else {
    return await MethodResults(slug);
  }
}