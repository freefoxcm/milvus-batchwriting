package org.vectors.milvus.actors;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.lang3.RandomStringUtils;
import org.nutz.lang.util.NutMap;
import akka.actor.UntypedAbstractActor;
import io.milvus.client.MilvusServiceClient;
import io.milvus.param.dml.InsertParam;

public class IndexActor extends UntypedAbstractActor {

    private MilvusServiceClient milvusClient;
    private String collectionName;
    private AtomicInteger doneActor;
    private AtomicInteger initActor;

    public IndexActor(MilvusServiceClient milvusClient, String collectionName, AtomicInteger initActor,
            AtomicInteger doneActor) {
        this.milvusClient = milvusClient;
        this.collectionName = collectionName;
        this.initActor = initActor;
        this.doneActor = doneActor;
    }

    @Override
    public void onReceive(Object message) throws Throwable {

        if (message instanceof NutMap) {

            long timeStep1 = System.currentTimeMillis();

            NutMap params = (NutMap) message;

            long startIndex = params.getLong("startIndex");
            int batchSize = params.getInt("batchSize");
            int dim = params.getInt("dim");

            Random ran = new Random();
            List<Long> list_ids = new ArrayList<>();
            List<String> list_tags = new ArrayList<>();
            List<List<Float>> list_features = new ArrayList<>();
            for (long i = startIndex; i < startIndex + batchSize; ++i) {
                list_ids.add(i);
                list_tags.add(RandomStringUtils.randomAlphanumeric(18));
                List<Float> vector = new ArrayList<>();
                for (int k = 0; k < dim; ++k) {
                    vector.add(ran.nextFloat());
                }
                list_features.add(vector);
            }

            long timeStep2 = System.currentTimeMillis();

            List<InsertParam.Field> fields = new ArrayList<>();
            fields.add(new InsertParam.Field("id", list_ids));
            fields.add(new InsertParam.Field("tags", list_tags));
            fields.add(new InsertParam.Field("feature", list_features));

            InsertParam insertParam = InsertParam.newBuilder()
                    .withCollectionName(collectionName)
                    .withFields(fields)
                    .build();

            milvusClient.insert(insertParam);

            doneActor.incrementAndGet();

            System.out.println(String.format("Acotr(%s/%s) %s-%s 创建数据集 %fs 写入数据集 %fs. ",
                    doneActor.get(),
                    initActor.get(),
                    startIndex + 1,
                    startIndex + batchSize,
                    (timeStep2 - timeStep1) / 1000F,
                    (System.currentTimeMillis() - timeStep1) / 1000F));

            getContext().stop(getSelf());
        } else {
            unhandled(message);
        }

    }
}