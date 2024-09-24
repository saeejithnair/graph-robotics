import { SceneNode } from '../types';

const createNode = (id: string, label: string, position: [number, number, number]): SceneNode => ({
  id,
  label,
  description: `Description for ${label}`,
  position,
  children: [],
});

export const generateMockSceneGraph = (): SceneNode => {
  const root = createNode('root', 'Scene', [0, 0, 0]);

  // Create a deep hierarchy
  const house = createNode('house', 'House', [0, 0, 0]);
  root.children.push(house);

  const floors = ['Ground Floor', 'First Floor', 'Attic'];
  floors.forEach((floorName, floorIndex) => {
    const floor = createNode(`floor-${floorIndex}`, floorName, [0, floorIndex * 3, 0]);
    house.children.push(floor);

    const rooms = ['Living Room', 'Kitchen', 'Bedroom', 'Bathroom'];
    rooms.forEach((roomName, roomIndex) => {
      const room = createNode(`${floorName.toLowerCase().replace(' ', '-')}-${roomName.toLowerCase().replace(' ', '-')}`, roomName, [roomIndex * 2, floorIndex * 3, 0]);
      floor.children.push(room);

      // Add detailed objects to each room
      switch (roomName) {
        case 'Living Room':
          addLivingRoomObjects(room);
          break;
        case 'Kitchen':
          addKitchenObjects(room);
          break;
        case 'Bedroom':
          addBedroomObjects(room);
          break;
        case 'Bathroom':
          addBathroomObjects(room);
          break;
      }
    });
  });

  return root;
};

function addLivingRoomObjects(room: SceneNode) {
  const sofa = createNode('sofa', 'Sofa', [0.5, 0, 0.5]);
  room.children.push(sofa);
  ['Cushion 1', 'Cushion 2', 'Throw Blanket'].forEach((item, index) => {
    sofa.children.push(createNode(`sofa-${item.toLowerCase().replace(' ', '-')}`, item, [0.5 + index * 0.2, 0.1, 0.5]));
  });

  const tvStand = createNode('tv-stand', 'TV Stand', [-0.5, 0, -0.5]);
  room.children.push(tvStand);
  const tv = createNode('tv', 'Television', [-0.5, 0.5, -0.5]);
  tvStand.children.push(tv);
  ['Remote', 'Game Console', 'DVD Player'].forEach((item, index) => {
    tvStand.children.push(createNode(`${item.toLowerCase().replace(' ', '-')}`, item, [-0.3 + index * 0.3, 0.1, -0.4]));
  });

  const coffeeTable = createNode('coffee-table', 'Coffee Table', [0, 0, 0]);
  room.children.push(coffeeTable);
  ['Magazine', 'Coaster', 'Vase'].forEach((item, index) => {
    coffeeTable.children.push(createNode(`table-${item.toLowerCase()}`, item, [0.1 * index, 0.1, 0.1 * index]));
  });
}

function addKitchenObjects(room: SceneNode) {
  const diningTable = createNode('dining-table', 'Dining Table', [0, 0, 0]);
  room.children.push(diningTable);

  for (let i = 0; i < 4; i++) {
    const placeSetting = createNode(`place-setting-${i}`, `Place Setting ${i + 1}`, [0.3 * i - 0.45, 0.1, 0.3 * (i % 2) - 0.15]);
    diningTable.children.push(placeSetting);

    const plate = createNode(`plate-${i}`, 'Plate', [0, 0.01, 0]);
    placeSetting.children.push(plate);
    ['Fork', 'Knife', 'Spoon'].forEach((utensil, index) => {
      plate.children.push(createNode(`${utensil.toLowerCase()}-${i}`, utensil, [0.05 * index - 0.05, 0.02, 0]));
    });

    const glass = createNode(`glass-${i}`, 'Glass', [0.1, 0.02, 0.1]);
    placeSetting.children.push(glass);
    glass.children.push(createNode(`drink-${i}`, 'Drink', [0, 0.01, 0]));
  }

  const kitchenIsland = createNode('kitchen-island', 'Kitchen Island', [-1, 0, 0]);
  room.children.push(kitchenIsland);
  ['Cutting Board', 'Knife Block', 'Fruit Bowl'].forEach((item, index) => {
    kitchenIsland.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [-1 + 0.2 * index, 0.1, 0]));
  });
}

function addBedroomObjects(room: SceneNode) {
  const bed = createNode('bed', 'Bed', [0, 0, 0]);
  room.children.push(bed);
  ['Pillow 1', 'Pillow 2', 'Blanket', 'Sheets'].forEach((item, index) => {
    bed.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [0.2 * index - 0.3, 0.1, 0]));
  });

  const dresser = createNode('dresser', 'Dresser', [-1, 0, -1]);
  room.children.push(dresser);
  ['Mirror', 'Jewelry Box', 'Photo Frame'].forEach((item, index) => {
    dresser.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [-1 + 0.2 * index, 0.5, -1]));
  });

  const closet = createNode('closet', 'Closet', [1, 0, -1]);
  room.children.push(closet);
  ['Clothes Rod', 'Shoe Rack', 'Storage Box'].forEach((item, index) => {
    closet.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [1, 0.5 * index, -1]));
  });
}

function addBathroomObjects(room: SceneNode) {
  const sink = createNode('sink', 'Sink', [0, 0.5, 0]);
  room.children.push(sink);
  ['Faucet', 'Soap Dispenser', 'Toothbrush Holder'].forEach((item, index) => {
    sink.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [0.2 * index - 0.2, 0.1, 0]));
  });

  const bathtub = createNode('bathtub', 'Bathtub', [-1, 0, -1]);
  room.children.push(bathtub);
  ['Shower Head', 'Shower Curtain', 'Bath Mat'].forEach((item, index) => {
    bathtub.children.push(createNode(item.toLowerCase().replace(' ', '-'), item, [-1 + 0.3 * index, 0.5, -1]));
  });

  const toilet = createNode('toilet', 'Toilet', [1, 0, -1]);
  room.children.push(toilet);
  toilet.children.push(createNode('toilet-paper', 'Toilet Paper', [1.2, 0.3, -1]));
}
