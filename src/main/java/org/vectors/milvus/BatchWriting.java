package org.vectors.milvus;

import java.util.concurrent.atomic.AtomicInteger;

import org.nutz.lang.util.NutMap;
import org.vectors.milvus.actors.IndexActor;

import com.typesafe.config.ConfigFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import io.milvus.client.MilvusServiceClient;
import io.milvus.grpc.DataType;
import io.milvus.param.ConnectParam;
import io.milvus.param.IndexType;
import io.milvus.param.MetricType;
import io.milvus.param.collection.CreateCollectionParam;
import io.milvus.param.collection.DropCollectionParam;
import io.milvus.param.collection.FieldType;
import io.milvus.param.collection.HasCollectionParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.index.CreateIndexParam;

public class BatchWriting {

    private static final String host = "127.0.0.1";
    private static final int port = 19530;
    private static final String collectionName = "vectors_test";
    private static final long entitySize = 10000000;// 总数据量
    private static final int entityBatchSize = 1000;// 每个batch数据量
    private static final int entityDim = 1024;// 向量维度

    public static void main(String[] args) throws InterruptedException, ClassNotFoundException {

        long timeStep1 = System.currentTimeMillis();

        final MilvusServiceClient milvusClient = new MilvusServiceClient(
                ConnectParam.newBuilder()
                        .withHost(host)
                        .withPort(port)
                        .build());

        long timeStep2 = System.currentTimeMillis();

        if (milvusClient.hasCollection(HasCollectionParam.newBuilder().withCollectionName(collectionName).build())
                .getData()) {
            milvusClient.dropCollection(DropCollectionParam.newBuilder().withCollectionName(collectionName).build());
        }

        long timeStep3 = System.currentTimeMillis();

        FieldType id = FieldType.newBuilder()
                .withName("id")
                .withDataType(DataType.Int64)
                .withPrimaryKey(true)
                .withAutoID(false)
                .build();
        FieldType tags = FieldType.newBuilder()
                .withName("tags")
                .withDataType(DataType.VarChar)
                .withMaxLength(120)
                .build();
        FieldType features = FieldType.newBuilder()
                .withName("feature")
                .withDataType(DataType.FloatVector)
                .withDimension(entityDim)
                .build();
        CreateCollectionParam createCollectionReq = CreateCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .withDescription("face vectors indexs")
                .withShardsNum(3)
                .addFieldType(id)
                .addFieldType(tags)
                .addFieldType(features)
                .build();

        milvusClient.createCollection(createCollectionReq);

        long timeStep4 = System.currentTimeMillis();

        AtomicInteger initActor = new AtomicInteger(0);
        AtomicInteger doneActor = new AtomicInteger(0);

        ActorSystem actorSystem = ActorSystem.create("milvusJobHandler",
                ConfigFactory.load("akka.conf"));

        for (int i = 0; i < entitySize; i += entityBatchSize) {
            initActor.incrementAndGet();
            ActorRef queryActor = actorSystem.actorOf(
                    Props.create(IndexActor.class,
                            milvusClient,
                            collectionName,
                            initActor,
                            doneActor),
                    String.format("milvus_%s", initActor.get()));
            queryActor.tell(NutMap.NEW()
                    .setv("startIndex", i)
                    .setv("batchSize", entityBatchSize)
                    .setv("dim", entityDim),
                    ActorRef.noSender());
        }

        Thread.sleep(1000);

        while (doneActor.get() < initActor.get()) {
            // 啥也不干，就干等着
        }

        // 关闭 Actor系统
        actorSystem.terminate();

        long timeStep5 = System.currentTimeMillis();

        final IndexType INDEX_TYPE = IndexType.IVF_FLAT; // IndexType
        final String INDEX_PARAM = "{\"nlist\":1024}"; // ExtraParam

        milvusClient.createIndex(
                CreateIndexParam.newBuilder()
                        .withCollectionName(collectionName)
                        .withFieldName("feature")
                        .withIndexType(INDEX_TYPE)
                        .withMetricType(MetricType.L2)
                        .withExtraParam(INDEX_PARAM)
                        .withSyncMode(Boolean.FALSE)
                        .build());

        long timeStep6 = System.currentTimeMillis();

        milvusClient.loadCollection(
                LoadCollectionParam.newBuilder()
                        .withCollectionName(collectionName)
                        .build());

        long timeStep7 = System.currentTimeMillis();

        milvusClient.close();

        long timeStep8 = System.currentTimeMillis();

        System.out.println(String.format("EntitySzie %d", entitySize));

        System.out.println(String.format("BatchSzie %d", entityBatchSize));

        System.out.println(String.format("连接到Milvus %fs", (timeStep2 - timeStep1) / 1000F));

        System.out.println(String.format("清理Collection %fs", (timeStep3 - timeStep2) / 1000F));

        System.out.println(String.format("创建Collection %fs", (timeStep4 - timeStep3) / 1000F));

        System.out.println(String.format("写入数据 %fs", (timeStep5 - timeStep4) / 1000F));

        System.out.println(String.format("创建索引 %fs", (timeStep6 - timeStep5) / 1000F));

        System.out.println(String.format("加载索引 %fs", (timeStep7 - timeStep6) / 1000F));

        System.out.println(String.format("断开连接 %fs", (timeStep8 - timeStep7) / 1000F));

    }
}