import MainHeader from './components/MainHeader';
import { IconExternalLink } from '@tabler/icons-react';
import DatasetResultsTable from '@/app/components/DatasetResultsTable';
import Link from 'next/link';
import * as api from '@/utils/api';

export default async function Home() {
  const datasets = await api.getDatasets();
  const datasetsResults = await Promise.all(datasets.map(async ({ id }) => api.getDatasetData(id)));
  return (
    <main>
      <MainHeader />
      {datasetsResults.map((datasetResults) => (
        <section key={datasetResults.id} className="result-set-panel">
          <div className="result-set-panel__header section_header">
            <h2>
              <Link href={`/${datasetResults.id}`}>{datasetResults.name}&nbsp;<span className="text-blue"><IconExternalLink size={20} /></span></Link>
            </h2>
            <p className="text-justify">{datasetResults.description}</p>
          </div>
          <DatasetResultsTable 
            datasetId={datasetResults.id} 
            methodResults={datasetResults.methods} 
            metrics={datasetResults.metrics}
            scenes={datasetResults.scenes}
            defaultMetric={datasetResults.default_metric || datasetResults.metrics[0].id} />
        </section>
      ))}
      <section>
        <h2>Acknowledgements</h2>
        <p>
        We want to thank Brent Yi and the <a href="https://github.com/nerfstudio-project/nerfstudio">NerfStudio Team</a> for helpful discussions regarding the NerfStudio codebase and for releasing the Viser platform.
This work was supported by the Czech Science Foundation (GA&#268;R) EXPRO (grant no. 23-07973X), the Grant Agency of the Czech Technical University in Prague (grant no. SGS24/095/OHK3/2T/13), and by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254).</p>
      </section>
      <section>
        <h2>License</h2>
        <p>
The NerfBaselines project is licensed under the <a href="https://raw.githubusercontent.com/jkulhanek/nerfbaselines/main/LICENSE">MIT license</a>.
Each implemented method is licensed under the license provided by the authors of the method.
For the currently implemented methods, the following licenses apply:
<ul>
<li>NerfStudio: <a href="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE">Apache 2.0</a></li>
<li>Instant-NGP: <a href="{https://raw.githubusercontent.com/NVlabs/instant-ngp/master/LICENSE.txt">custom, research purposes only</a></li>
<li>Gaussian-Splatting: <a href="{https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md">custom, research purposes only</a></li>
<li>Mip-Splatting: <a href="{https://raw.githubusercontent.com/autonomousvision/mip-splatting/main/LICENSE.md">custom, research purposes only</a></li>
<li>Gaussian Opacity Fields: <a href="{https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/LICENSE.md">custom, research purposes only</a></li>
<li>Tetra-NeRF: <a href="{https://raw.githubusercontent.com/jkulhanek/tetra-nerf/master/LICENSE">MIT</a>, <a href="{https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE">Apache 2.0</a></li>
<li>Mip-NeRF 360: <a href="{https://raw.githubusercontent.com/google-research/multinerf/main/LICENSE">Apache 2.0</a></li>
<li>Zip-NeRF: <a href="{https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE">Apache 2.0</a></li>
<li>CamP: <a href="{https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE">Apache 2.0</a></li>
</ul>
        </p>
      </section>
    </main>
  );
}
// <Button variant="filled" radius="xl">github</Button>
